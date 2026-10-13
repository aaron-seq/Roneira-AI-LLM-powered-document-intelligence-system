"""Baseline: the schema as it stood when migrations were introduced.

This reproduces exactly what ``Base.metadata.create_all`` used to build, so a
database created by that bootstrap is at this revision already — startup
stamps such a database here rather than replaying this migration against
tables that exist. Every column default is Python-side in the models
(``default=`` not ``server_default=``), which is why no server defaults appear
below: anything else would describe a different schema than the one in the
field.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-10-12

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001_baseline"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chat_feedback",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("message_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=True),
        sa.Column("owner_id", sa.String(length=128), nullable=True),
        sa.Column("is_positive", sa.Boolean(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_chat_feedback_created_at"), "chat_feedback", ["created_at"], unique=False
    )
    op.create_index(
        op.f("ix_chat_feedback_message_id"), "chat_feedback", ["message_id"], unique=False
    )
    op.create_index(
        op.f("ix_chat_feedback_owner_id"), "chat_feedback", ["owner_id"], unique=False
    )
    op.create_index(
        op.f("ix_chat_feedback_session_id"), "chat_feedback", ["session_id"], unique=False
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=128), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("content_type", sa.String(length=128), nullable=True),
        sa.Column("file_extension", sa.String(length=16), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("checksum", sa.String(length=64), nullable=True),
        sa.Column("storage_path", sa.String(length=1024), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "QUEUED",
                "PROCESSING",
                "COMPLETED",
                "FAILED",
                name="documentstatus",
                native_enum=False,
                length=16,
            ),
            nullable=False,
        ),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("message", sa.String(length=1024), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("word_count", sa.Integer(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("extracted_text", sa.Text(), nullable=True),
        sa.Column("doc_metadata", sa.JSON(), nullable=False),
        sa.Column("ai_analysis", sa.JSON(), nullable=False),
        sa.Column("processing_options", sa.JSON(), nullable=False),
        sa.Column("embeddings_are_real", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_documents_checksum"), "documents", ["checksum"], unique=False
    )
    op.create_index(
        op.f("ix_documents_created_at"), "documents", ["created_at"], unique=False
    )
    op.create_index(
        "ix_documents_owner_created",
        "documents",
        ["owner_id", "created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_documents_owner_id"), "documents", ["owner_id"], unique=False
    )
    op.create_index(
        "ix_documents_owner_status", "documents", ["owner_id", "status"], unique=False
    )
    op.create_index(op.f("ix_documents_status"), "documents", ["status"], unique=False)

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.String(length=96), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=128), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("start_char", sa.Integer(), nullable=False),
        sa.Column("end_char", sa.Integer(), nullable=False),
        sa.Column("token_estimate", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_chunks_doc_index",
        "document_chunks",
        ["document_id", "chunk_index"],
        unique=False,
    )
    op.create_index(
        op.f("ix_document_chunks_document_id"),
        "document_chunks",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_document_chunks_owner_id"), "document_chunks", ["owner_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_document_chunks_owner_id"), table_name="document_chunks")
    op.drop_index(op.f("ix_document_chunks_document_id"), table_name="document_chunks")
    op.drop_index("ix_chunks_doc_index", table_name="document_chunks")

    op.drop_table("document_chunks")
    op.drop_index(op.f("ix_documents_status"), table_name="documents")
    op.drop_index("ix_documents_owner_status", table_name="documents")
    op.drop_index(op.f("ix_documents_owner_id"), table_name="documents")
    op.drop_index("ix_documents_owner_created", table_name="documents")
    op.drop_index(op.f("ix_documents_created_at"), table_name="documents")
    op.drop_index(op.f("ix_documents_checksum"), table_name="documents")

    op.drop_table("documents")
    op.drop_index(op.f("ix_chat_feedback_session_id"), table_name="chat_feedback")
    op.drop_index(op.f("ix_chat_feedback_owner_id"), table_name="chat_feedback")
    op.drop_index(op.f("ix_chat_feedback_message_id"), table_name="chat_feedback")
    op.drop_index(op.f("ix_chat_feedback_created_at"), table_name="chat_feedback")

    op.drop_table("chat_feedback")
