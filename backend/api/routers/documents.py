"""Document upload, status, retrieval and deletion endpoints.

Every route is scoped to the authenticated caller: a document is only visible
to its owner. Before ownership existed, ``GET /api/documents`` returned every
document uploaded by anyone.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse, PlainTextResponse

from backend.api.dependencies import get_document_processor, get_websocket_manager
from backend.api.security import CurrentUser, get_current_user
from backend.models.responses import (
    DocumentChange,
    DocumentComparisonResponse,
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from backend.services.document_comparison import compare_documents
from backend.services.document_export import (
    render_comparison_markdown,
    render_markdown,
)
from backend.services.document_processor import DocumentProcessorService
from backend.services.file_validation import (
    ALLOWED_EXTENSIONS,
    FileValidationError,
)
from backend.services.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for processing",
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to process"),
    extract_tables: bool = Form(default=True),
    extract_images: bool = Form(default=False),
    enhance_with_ai: bool = Form(default=True),
    user: CurrentUser = Depends(get_current_user),
    processor: DocumentProcessorService = Depends(get_document_processor),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager),
) -> DocumentUploadResponse:
    """Accept a document, store it, and queue it for processing.

    Returns 202 with a ``document_id``; poll ``/{document_id}/status`` or
    subscribe over the websocket for progress.
    """
    options: Dict[str, Any] = {
        "extract_tables": extract_tables,
        "extract_images": extract_images,
        "enhance_with_ai": enhance_with_ai,
    }

    try:
        document_id, stored = await processor.accept_upload(
            file=file, owner_id=user.user_id, processing_options=options
        )
    except FileValidationError as exc:
        # Validation messages are written for the user and safe to return.
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:
        logger.exception("Upload failed for user %s", user.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document upload failed. Please try again.",
        ) from exc

    background_tasks.add_task(
        _process_document_task,
        processor,
        websocket_manager,
        document_id,
        user.user_id,
        stored.path,
        options,
        os.path.basename(file.filename),
    )

    return DocumentUploadResponse(
        document_id=document_id,
        status="queued",
        message="Document accepted and queued for processing",
        filename=os.path.basename(file.filename),
        size_bytes=stored.size_bytes,
        checksum=stored.checksum,
        detected_type=stored.detected_type,
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List your documents",
)
async def list_documents(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status_filter: Optional[str] = Query(
        default=None,
        description="Filter by status: queued, processing, completed, failed",
    ),
    user: CurrentUser = Depends(get_current_user),
    processor: DocumentProcessorService = Depends(get_document_processor),
) -> DocumentListResponse:
    """Return a page of the caller's documents, newest first."""
    try:
        documents, total = await processor.list_documents(
            limit=limit,
            offset=offset,
            status_filter=status_filter,
            owner_id=user.user_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return DocumentListResponse(
        documents=[DocumentStatusResponse.model_validate(d) for d in documents],
        total=total,
        limit=limit,
        offset=offset,
        has_more=offset + len(documents) < total,
    )


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Get processing status",
)
async def get_document_status(
    document_id: str,
    user: CurrentUser = Depends(get_current_user),
    processor: DocumentProcessorService = Depends(get_document_processor),
) -> DocumentStatusResponse:
    """Return the current processing state of one document."""
    status_data = await processor.get_document_status(document_id, owner_id=user.user_id)
    if not status_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )
    return DocumentStatusResponse(**status_data)


@router.get(
    "/compare",
    response_model=DocumentComparisonResponse,
    summary="Compare two documents paragraph by paragraph",
)
async def compare(
    left: str = Query(description="Document ID of the baseline document"),
    right: str = Query(description="Document ID of the document to compare against"),
    fmt: str = Query(
        default="json",
        pattern="^(json|markdown)$",
        description="'markdown' returns a downloadable report instead of JSON",
    ),
    user: CurrentUser = Depends(get_current_user),
    processor: DocumentProcessorService = Depends(get_document_processor),
):
    """Report what changed between two of the caller's documents.

    The answer is computed from the stored text of both documents, so it does
    not depend on the LLM and cannot invent a change that is not there.

    Declared above ``/{document_id}`` deliberately: FastAPI matches routes in
    order, so a literal path registered later would be swallowed by the
    parameterised one and "compare" would be looked up as a document ID.
    """
    documents = {}
    for side, document_id in (("left", left), ("right", right)):
        # Ownership is enforced by passing owner_id, so one caller cannot
        # diff another tenant's document — or learn that it exists.
        document = await processor.repository.get(document_id, owner_id=user.user_id)
        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}",
            )
        if not (document.extracted_text or "").strip():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Document {document_id} has no extracted text to compare. "
                    f"Its status is '{document.status}'."
                ),
            )
        documents[side] = document

    result = compare_documents(
        documents["left"].extracted_text, documents["right"].extracted_text
    )
    result["left_filename"] = documents["left"].filename
    result["right_filename"] = documents["right"].filename

    if fmt == "markdown":
        return PlainTextResponse(
            render_comparison_markdown(result),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="comparison.md"'},
        )

    return DocumentComparisonResponse(
        left_document_id=left,
        right_document_id=right,
        left_filename=documents["left"].filename,
        right_filename=documents["right"].filename,
        changes=[DocumentChange(**change) for change in result["changes"]],
        added=result["added"],
        removed=result["removed"],
        changed=result["changed"],
        unchanged_paragraphs=result["unchanged_paragraphs"],
        left_paragraphs=result["left_paragraphs"],
        right_paragraphs=result["right_paragraphs"],
        similarity=result["similarity"],
        truncated=result["truncated"],
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get a document with its extracted text and analysis",
)
async def get_document(
    document_id: str,
    user: CurrentUser = Depends(get_current_user),
    processor: DocumentProcessorService = Depends(get_document_processor),
) -> DocumentDetailResponse:
    """Return the full processing result for one document."""
    document = await processor.get_document(document_id, owner_id=user.user_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )
    return DocumentDetailResponse(**document)


@router.get(
    "/{document_id}/export",
    summary="Export a document's analysis as Markdown",
    response_class=PlainTextResponse,
)
async def export_document(
    document_id: str,
    user: CurrentUser = Depends(get_current_user),
    processor: DocumentProcessorService = Depends(get_document_processor),
) -> PlainTextResponse:
    """Return the summary, extracted details and text as a Markdown file.

    Personal data is reported as counts only, never as values — the same rule
    the rest of the API follows. An export is the most likely thing to be
    pasted into a ticket, which is the worst place for a card number.
    """
    document = await processor.get_document(document_id, owner_id=user.user_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    filename = os.path.basename(document.get("filename") or document_id)
    return PlainTextResponse(
        render_markdown(document),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}.md"',
        },
    )


@router.get(
    "/{document_id}/source",
    summary="Download the original uploaded file",
    response_class=FileResponse,
)
async def download_source(
    document_id: str,
    user: CurrentUser = Depends(get_current_user),
    processor: DocumentProcessorService = Depends(get_document_processor),
):
    """Return the original file so a citation can be verified against it.

    Serving this through an authorized route (rather than a public static
    mount) is what keeps one tenant's documents out of another's reach.
    """
    document = await processor.repository.get(document_id, owner_id=user.user_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )
    if not document.storage_path or not os.path.exists(document.storage_path):
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=(
                "The source file is no longer retained for this document. "
                "Source retention is controlled by RETAIN_SOURCE_FILES and "
                "SOURCE_RETENTION_DAYS."
            ),
        )

    return FileResponse(
        path=document.storage_path,
        filename=document.filename,
        media_type=document.content_type or "application/octet-stream",
        # Never render an uploaded document inline in the browser: an HTML or
        # SVG payload served same-origin would run as our own page.
        content_disposition_type="attachment",
    )


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document and its index entries",
)
async def delete_document(
    document_id: str,
    user: CurrentUser = Depends(get_current_user),
    processor: DocumentProcessorService = Depends(get_document_processor),
) -> DocumentDeleteResponse:
    """Remove a document, its chunks, its vectors, and its stored file."""
    deleted = await processor.delete_document(document_id, owner_id=user.user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )
    return DocumentDeleteResponse(document_id=document_id, deleted=True)


@router.get("/formats/supported", summary="List accepted upload formats")
async def supported_formats() -> Dict[str, Any]:
    """Report what the server will accept, so clients need not guess."""
    from backend.core.config import get_settings

    settings = get_settings()
    return {
        "extensions": sorted(ALLOWED_EXTENSIONS),
        "max_file_size_bytes": settings.max_file_size,
        "max_file_size_mb": settings.max_file_size // (1024 * 1024),
    }


@router.websocket("/ws/{client_id}")
async def websocket_document_updates(
    websocket: WebSocket,
    client_id: str,
    websocket_manager: WebSocketManager = Depends(get_websocket_manager),
):
    """Stream processing progress to a connected client."""
    try:
        await websocket_manager.connect(websocket, client_id)
        while True:
            message = await websocket.receive_text()
            await websocket_manager.handle_client_message(client_id, message)
    except WebSocketDisconnect:
        await websocket_manager.disconnect(client_id)
    except Exception as exc:
        logger.warning("WebSocket error for client %s: %s", client_id, exc)
        await websocket_manager.disconnect(client_id)


async def _process_document_task(
    processor: DocumentProcessorService,
    ws_manager: WebSocketManager,
    document_id: str,
    owner_id: str,
    file_path: str,
    options: Dict[str, Any],
    filename: str,
) -> None:
    """Background entry point for the processing pipeline."""

    async def progress(update: Dict[str, Any]) -> None:
        # broadcast_progress takes a percentage and a stage label, not the
        # whole status dict; passing the dict put an object where clients
        # expect a number.
        await ws_manager.broadcast_progress(
            document_id,
            progress=int(update.get("progress", 0)),
            stage=update.get("message"),
        )
        if update.get("status") in ("completed", "failed"):
            await ws_manager.broadcast_status(document_id, update["status"], data=update)

    await processor.process_document(
        document_id=document_id,
        owner_id=owner_id,
        file_path=file_path,
        options=options,
        filename=filename,
        progress_callback=progress,
    )
