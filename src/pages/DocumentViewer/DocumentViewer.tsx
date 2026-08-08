import { useState, useEffect, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import {
    Box,
    Typography,
    Paper,
    TextField,
    Button,
    CircularProgress,
    Grid,
    Chip,
    Divider,
    Alert,
    MenuItem,
    Select,
    Stack,
    Tooltip,
} from '@mui/material';
import {
    Download,
    Description,
    DocumentScanner,
    PrivacyTip,
    WarningAmber,
    CompareArrows,
} from '@mui/icons-material';
import { apiClient } from '../../api/client';
import toast from 'react-hot-toast';

// Mirrors GET /api/documents/{id}. `result` is only present on this endpoint —
// the previous version fetched /status, which never carries it, so the
// extracted text panel always read "No text extracted yet."
interface DocumentDetail {
    document_id: string;
    status: string;
    filename?: string | null;
    error?: string | null;
    page_count?: number | null;
    word_count?: number | null;
    chunk_count?: number;
    embeddings_are_real?: boolean;
    ai_insights?: {
        summary?: string;
        summary_source?: string;
        entities?: Record<string, string[]>;
        pii?: { found: number; confident: number; by_type: Record<string, number> };
    } | null;
    result?: {
        original_text?: string;
        metadata?: Record<string, any>;
    } | null;
}

interface ComparisonChange {
    kind: string;
    left: string[];
    right: string[];
    left_page?: number | null;
    right_page?: number | null;
}

/** "1 pages" is the sort of thing that makes an interface look unfinished. */
const plural = (count: number, noun: string) =>
    `${count} ${noun}${count === 1 ? '' : 's'}`;

const CHANGE_COLOURS: Record<string, string> = {
    added: '#10b981',
    removed: '#ef4444',
    changed: '#f59e0b',
};

const DocumentViewer = () => {
    const { documentId } = useParams();
    const [documentData, setDocumentData] = useState<DocumentDetail | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [question, setQuestion] = useState('');
    const [chatHistory, setChatHistory] = useState<any[]>([]);
    const [isAnswering, setIsAnswering] = useState(false);

    const [otherDocuments, setOtherDocuments] = useState<DocumentDetail[]>([]);
    const [compareWith, setCompareWith] = useState('');
    const [comparison, setComparison] = useState<any>(null);
    const [isComparing, setIsComparing] = useState(false);

    const fetchDocument = useCallback(async () => {
        try {
            const response = await apiClient.get(`/documents/${documentId}`);
            setDocumentData(response.data);
        } catch {
            toast.error('Failed to load this document.');
        } finally {
            setIsLoading(false);
        }
    }, [documentId]);

    useEffect(() => {
        if (documentId) fetchDocument();
    }, [documentId, fetchDocument]);

    useEffect(() => {
        apiClient
            .get('/documents?limit=100')
            .then((response) =>
                setOtherDocuments(
                    (response.data?.documents ?? []).filter(
                        (d: DocumentDetail) =>
                            d.document_id !== documentId && d.status === 'completed',
                    ),
                ),
            )
            .catch(() => undefined);
    }, [documentId]);

    // Both downloads go through apiClient rather than a plain href or
    // window.open: these endpoints require a bearer token, and a browser
    // navigation sends no Authorization header, so the link would just 401.
    const saveBlob = (data: BlobPart, filename: string) => {
        const url = window.URL.createObjectURL(new Blob([data]));
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.setTimeout(() => window.URL.revokeObjectURL(url), 0);
    };

    const handleDownload = async () => {
        try {
            const response = await apiClient.get(`/documents/${documentId}/source`, {
                responseType: 'blob',
            });
            saveBlob(response.data, documentData?.filename || String(documentId));
        } catch {
            toast.error('The source file is no longer retained for this document.');
        }
    };

    const handleExport = async () => {
        try {
            const response = await apiClient.get(`/documents/${documentId}/export`, {
                responseType: 'blob',
            });
            saveBlob(response.data, `${documentData?.filename || documentId}.md`);
        } catch {
            toast.error('Could not export this document.');
        }
    };

    const handleCompare = async () => {
        if (!compareWith) return;
        setIsComparing(true);
        try {
            const response = await apiClient.get('/documents/compare', {
                params: { left: documentId, right: compareWith },
            });
            setComparison(response.data);
        } catch {
            toast.error('Could not compare these documents.');
        } finally {
            setIsComparing(false);
        }
    };

    const handleAskQuestion = async () => {
        if (!question.trim()) return;
        setIsAnswering(true);

        const newHistory = [...chatHistory, { role: 'user', content: question }];
        setChatHistory(newHistory);

        try {
            const response = await apiClient.post('/chat', {
                message: question,
                session_id: `doc-${documentId}`,
                use_rag: true,
                rag_top_k: 3,
                document_id: documentId,
                detailed: false,
            });

            setChatHistory([
                ...newHistory,
                {
                    role: 'assistant',
                    content: response.data.message,
                    grounded: response.data.grounded,
                },
            ]);
            setQuestion('');
        } catch {
            toast.error('Failed to get an answer.');
            setChatHistory(newHistory);
        } finally {
            setIsAnswering(false);
        }
    };

    if (isLoading) return <CircularProgress />;

    if (!documentData) {
        return <Alert severity="error">This document could not be loaded.</Alert>;
    }

    const metadata = documentData.result?.metadata ?? {};
    const insights = documentData.ai_insights ?? {};
    const ocrPages: number[] = metadata.ocr_pages ?? [];
    const pii = insights.pii;
    const entities = insights.entities ?? {};
    const text = documentData.result?.original_text;

    return (
        <Grid container spacing={3}>
            <Grid item xs={12}>
                <Typography variant="h5">
                    {documentData.filename || 'Document'}
                </Typography>

                <Stack direction="row" spacing={1} sx={{ mt: 1, flexWrap: 'wrap', gap: 1 }}>
                    <Chip
                        size="small"
                        label={documentData.status}
                        color={documentData.status === 'completed' ? 'success' : 'default'}
                    />
                    {documentData.page_count ? (
                        <Chip size="small" variant="outlined" label={plural(documentData.page_count, 'page')} />
                    ) : null}
                    {documentData.word_count ? (
                        <Chip size="small" variant="outlined" label={plural(documentData.word_count, 'word')} />
                    ) : null}
                    <Chip
                        size="small"
                        variant="outlined"
                        label={plural(documentData.chunk_count ?? 0, 'searchable chunk')}
                    />

                    {/* Machine-read pages are worth flagging: OCR quality is
                        limited by the scan, and this text is what gets cited. */}
                    {ocrPages.length > 0 && (
                        <Tooltip title="These pages had no text layer and were read with OCR. Check them against the original before relying on an answer drawn from them.">
                            <Chip
                                size="small"
                                color="info"
                                variant="outlined"
                                icon={<DocumentScanner sx={{ fontSize: 16 }} />}
                                label={`OCR: page${ocrPages.length > 1 ? 's' : ''} ${ocrPages.join(', ')}`}
                            />
                        </Tooltip>
                    )}

                    {documentData.embeddings_are_real === false && (
                        <Tooltip title="No embedding model was loaded when this document was indexed, so it is matched on keywords rather than meaning.">
                            <Chip
                                size="small"
                                color="warning"
                                variant="outlined"
                                icon={<WarningAmber sx={{ fontSize: 16 }} />}
                                label="Keyword-only index"
                            />
                        </Tooltip>
                    )}

                    <Button
                        size="small"
                        startIcon={<Download />}
                        onClick={handleDownload}
                        sx={{ ml: 'auto' }}
                    >
                        Original
                    </Button>
                    <Button
                        size="small"
                        startIcon={<Description />}
                        onClick={handleExport}
                    >
                        Export
                    </Button>
                </Stack>

                {documentData.error && (
                    <Alert severity="error" sx={{ mt: 2 }}>
                        {documentData.error}
                    </Alert>
                )}

                {pii && pii.found > 0 && (
                    <Alert
                        severity="warning"
                        icon={<PrivacyTip />}
                        sx={{ mt: 2 }}
                    >
                        Possible personal data found:{' '}
                        {Object.entries(pii.by_type)
                            .map(([type, count]) => `${count} x ${type.toLowerCase().replace(/_/g, ' ')}`)
                            .join(', ')}
                        . Detected by pattern matching, so treat it as a prompt to check, not a verdict.
                    </Alert>
                )}
            </Grid>

            <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2, height: '70vh', overflowY: 'auto' }}>
                    {insights.summary && (
                        <>
                            <Typography variant="h6">Summary</Typography>
                            <Typography variant="caption" color="text.secondary">
                                {insights.summary_source === 'extractive'
                                    ? "The document's own sentences, selected automatically — no model was involved."
                                    : 'Written by the language model from the document.'}
                            </Typography>
                            <Typography paragraph sx={{ mt: 1 }}>
                                {insights.summary}
                            </Typography>
                            <Divider sx={{ my: 2 }} />
                        </>
                    )}

                    {Object.keys(entities).length > 0 && (
                        <>
                            <Typography variant="h6">Extracted details</Typography>
                            <Box sx={{ mt: 1 }}>
                                {Object.entries(entities).map(([type, values]) => (
                                    <Box key={type} sx={{ mb: 1 }}>
                                        <Typography variant="caption" color="text.secondary">
                                            {type}
                                        </Typography>
                                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                                            {values.map((value) => (
                                                <Chip key={`${type}-${value}`} size="small" label={value} />
                                            ))}
                                        </Box>
                                    </Box>
                                ))}
                            </Box>
                            <Divider sx={{ my: 2 }} />
                        </>
                    )}

                    <Typography variant="h6">Extracted text</Typography>
                    <Typography
                        paragraph
                        sx={{ whiteSpace: 'pre-wrap', fontSize: '0.85rem', mt: 1 }}
                    >
                        {text && text.trim()
                            ? text
                            : 'No text was extracted from this document.'}
                    </Typography>
                </Paper>
            </Grid>

            <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2, height: '70vh', display: 'flex', flexDirection: 'column' }}>
                    <Typography variant="h6" gutterBottom>
                        Ask about this document
                    </Typography>
                    <Box sx={{ flexGrow: 1, overflowY: 'auto', mb: 2 }}>
                        {chatHistory.length === 0 && (
                            <Typography variant="body2" color="text.secondary">
                                Answers here are drawn only from this document, and say so when
                                nothing in it supports the question.
                            </Typography>
                        )}
                        {chatHistory.map((entry, index) => (
                            <Box
                                key={index}
                                sx={{ mb: 1, textAlign: entry.role === 'user' ? 'right' : 'left' }}
                            >
                                <Typography variant="caption" color="text.secondary">
                                    {entry.role}
                                </Typography>
                                <Paper sx={{ p: 1, display: 'inline-block' }}>{entry.content}</Paper>
                                {entry.role === 'assistant' && entry.grounded === false && (
                                    <Typography variant="caption" display="block" color="warning.main">
                                        Not supported by this document
                                    </Typography>
                                )}
                            </Box>
                        ))}
                    </Box>
                    <Box sx={{ display: 'flex' }}>
                        <TextField
                            fullWidth
                            size="small"
                            label="Ask a question about the document"
                            value={question}
                            onChange={(e) => setQuestion(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && handleAskQuestion()}
                        />
                        <Button
                            variant="contained"
                            onClick={handleAskQuestion}
                            disabled={isAnswering}
                            sx={{ ml: 1 }}
                        >
                            {isAnswering ? <CircularProgress size={24} /> : 'Ask'}
                        </Button>
                    </Box>
                </Paper>
            </Grid>

            <Grid item xs={12}>
                <Paper sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>
                        Compare with another document
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                        Differences are computed from the text of both documents, so nothing
                        here is generated.
                    </Typography>

                    <Stack direction="row" spacing={1} sx={{ mt: 2, alignItems: 'center' }}>
                        <Select
                            size="small"
                            displayEmpty
                            value={compareWith}
                            onChange={(e) => setCompareWith(e.target.value)}
                            sx={{ minWidth: 260 }}
                        >
                            <MenuItem value="">
                                <em>Choose a document…</em>
                            </MenuItem>
                            {otherDocuments.map((doc) => (
                                <MenuItem key={doc.document_id} value={doc.document_id}>
                                    {doc.filename || doc.document_id}
                                </MenuItem>
                            ))}
                        </Select>
                        <Button
                            variant="outlined"
                            startIcon={<CompareArrows />}
                            onClick={handleCompare}
                            disabled={!compareWith || isComparing}
                        >
                            {isComparing ? <CircularProgress size={20} /> : 'Compare'}
                        </Button>
                    </Stack>

                    {otherDocuments.length === 0 && (
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                            Upload a second document to compare this one against.
                        </Typography>
                    )}

                    {comparison && (
                        <Box sx={{ mt: 2 }}>
                            <Stack direction="row" spacing={1} sx={{ mb: 2, flexWrap: 'wrap', gap: 1 }}>
                                <Chip size="small" label={`${Math.round(comparison.similarity * 100)}% unchanged`} />
                                <Chip size="small" label={`${comparison.changed} changed`} />
                                <Chip size="small" label={`${comparison.added} added`} />
                                <Chip size="small" label={`${comparison.removed} removed`} />
                            </Stack>

                            {comparison.changes.length === 0 && (
                                <Alert severity="success">
                                    These documents are identical, paragraph for paragraph.
                                </Alert>
                            )}

                            {comparison.changes.map((change: ComparisonChange, index: number) => (
                                <Box
                                    key={index}
                                    sx={{
                                        mb: 1.5,
                                        pl: 1.5,
                                        borderLeft: `3px solid ${CHANGE_COLOURS[change.kind] ?? '#888'}`,
                                    }}
                                >
                                    <Typography variant="caption" sx={{ color: CHANGE_COLOURS[change.kind] }}>
                                        {change.kind}
                                        {change.left_page ? ` — page ${change.left_page}` : ''}
                                    </Typography>
                                    {change.left.map((paragraph, i) => (
                                        <Typography
                                            key={`l-${i}`}
                                            variant="body2"
                                            sx={{ textDecoration: 'line-through', opacity: 0.7 }}
                                        >
                                            {paragraph}
                                        </Typography>
                                    ))}
                                    {change.right.map((paragraph, i) => (
                                        <Typography key={`r-${i}`} variant="body2">
                                            {paragraph}
                                        </Typography>
                                    ))}
                                </Box>
                            ))}

                            {comparison.truncated && (
                                <Alert severity="info">
                                    Only the first {comparison.changes.length} changes are shown.
                                </Alert>
                            )}
                        </Box>
                    )}
                </Paper>
            </Grid>
        </Grid>
    );
};

export default DocumentViewer;
