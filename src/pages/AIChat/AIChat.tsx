import { useState, useRef, useEffect } from 'react';
import { 
    Box, 
    TextField, 
    Typography, 
    IconButton, 
    Paper,
    Chip,
    CircularProgress,
    Tooltip,
    Collapse,
    Button,
    Drawer,
    Slider,
    Switch,
    FormControlLabel,
    Divider
} from '@mui/material';
import { 
    Send, 
    AttachFile, 
    AutoAwesome,
    Description,
    ExpandMore,
    ExpandLess,
    ContentCopy,
    ThumbUp,
    ThumbDown,
    Refresh,
    Settings,
    ArrowBack,
    Download,
    VerifiedUser,
    WarningAmber
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { apiClient } from '../../api/client';

// ==============================================================================
// Roneira AI - AI Chat Interface
// Document Intelligence with RAG (Retrieval Augmented Generation)
// ==============================================================================

// Mirrors the `sources[]` entries of POST /api/chat exactly. The previous
// shape ({ id, relevance, excerpt }) matched no field the API returns, so
// every citation rendered an empty excerpt and every download requested
// /documents/undefined/source. axios types responses as `any`, so nothing
// failed at compile time — only in front of a user.
interface DocumentReference {
    document_id: string;
    chunk_id: string;
    filename: string | null;
    page_number: number | null;
    score: number;
    content_preview: string;
}

interface ChatMessage {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    references?: DocumentReference[];
    isLoading?: boolean;
    isDetailed?: boolean;
    // Honesty signals from the API. `grounded` is false when the answer is not
    // supported by the user's documents; `embeddingsAreReal` is false when
    // search has degraded to keyword matching. Both were previously dropped on
    // the floor, so a degraded answer looked identical to a trustworthy one.
    grounded?: boolean;
    embeddingsAreReal?: boolean;
}

const AIChat = () => {
    const navigate = useNavigate();
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [documents, setDocuments] = useState<any[]>([]);
    const [detailedMode, setDetailedMode] = useState(false);
    const [expandedRefs, setExpandedRefs] = useState<string[]>([]);
    const [sessionId] = useState(() => crypto.randomUUID());
    const [settingsOpen, setSettingsOpen] = useState(false);
    const [ragEnabled, setRagEnabled] = useState(true);
    const [temperature, setTemperature] = useState(0.7);
    const [maxTokens, setMaxTokens] = useState(512);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    // Initial Welcome Message
    useEffect(() => {
        setMessages([{
            id: 'welcome',
            role: 'assistant',
            content: "Hello! This is Roneira AI, ready to help you with searching information or specifying documents you need. I can analyze your invoices, HR policies, and engineering specs to provide accurate answers with citations.",
            timestamp: new Date()
        }]);
        fetchDocuments();
    }, []);

    // Auto-scroll to bottom
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const fetchDocuments = async () => {
        try {
            // Correct endpoint for document list
            const response = await apiClient.get('/documents?limit=100');
            console.log('Fetched documents:', response.data);
            if (response.data && response.data.documents) {
                setDocuments(response.data.documents);
            }
        } catch (error) {
            console.error('Failed to fetch documents:', error);
        }
    };

    // Download the source document behind a citation.
    //
    // This cannot use window.open: the endpoint requires a bearer token, and
    // a plain navigation sends no Authorization header (the previous version
    // also pointed at /download, which does not exist — the route is
    // /api/documents/{id}/source). Fetching as a blob keeps the header and
    // gives the browser a correct filename.
    const handleDownload = async (docId: string, filename: string) => {
        try {
            const response = await apiClient.get(`/documents/${docId}/source`, {
                responseType: 'blob',
            });

            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.download = filename || `${docId}`;
            document.body.appendChild(link);
            link.click();
            link.remove();
            // Revoke on the next tick so the download has started.
            window.setTimeout(() => window.URL.revokeObjectURL(url), 0);
        } catch {
            toast.error(
                'Could not download the source document. It may no longer be retained.',
            );
        }
    };

    // Query LLM with document context
    const queryLLM = async (query: string, isDetailed: boolean) => {
        try {
            // Correct endpoint for chat completion
            const response = await apiClient.post('/chat', {
                message: query,
                session_id: sessionId,
                use_rag: true,
                rag_top_k: 3,
                document_id: null,
                max_tokens: isDetailed ? 2048 : 512,
                detailed: isDetailed
            });

            if (response.status === 200) {
                return response.data;
            }
            throw new Error('Query failed');
        } catch (error) {
            console.error('LLM query failed:', error);
            return null;
        }
    };

    // Handle sending message
    const handleSend = async () => {
        if (!inputValue.trim() || isLoading) return;
        
        const userMessage: ChatMessage = {
            id: Date.now().toString(),
            role: 'user',
            content: inputValue,
            timestamp: new Date()
        };
        
        const assistantMessage: ChatMessage = {
            id: (Date.now() + 1).toString(),
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            isLoading: true,
            isDetailed: detailedMode
        };
        
        setMessages(prev => [...prev, userMessage, assistantMessage]);
        setInputValue('');
        setIsLoading(true);
        
        try {
            const data = await queryLLM(inputValue, detailedMode);
            
            if (data) {
                setMessages(prev => prev.map(msg => 
                    msg.id === assistantMessage.id 
                        ? {
                            ...msg,
                            content: data.message,
                            isLoading: false,
                            references: data.sources,
                            grounded: data.grounded,
                            embeddingsAreReal: data.embeddings_are_real,
                        }
                        : msg
                ));
            } else {
                setMessages(prev => prev.map(msg => 
                    msg.id === assistantMessage.id 
                        ? { ...msg, content: "I encountered an error connecting to the LLM service. Please try again.", isLoading: false }
                        : msg
                ));
            }
        } catch (error) {
            setMessages(prev => prev.map(msg =>
                msg.id === assistantMessage.id
                    ? { ...msg, content: 'An error occurred while processing your request.', isLoading: false }
                    : msg
            ));
        } finally {
            setIsLoading(false);
        }
    };

    const toggleRefExpand = (id: string) => {
        setExpandedRefs(prev => 
            prev.includes(id) ? prev.filter(r => r !== id) : [...prev, id]
        );
    };

    const copyToClipboard = (text: string) => {
        navigator.clipboard.writeText(text);
        toast.success('Copied to clipboard');
    };

    // Regenerate response for a specific message
    const regenerateResponse = async (messageId: string) => {
        // Find the user message that preceded this assistant message
        const messageIndex = messages.findIndex(m => m.id === messageId);
        if (messageIndex <= 0) return;
        
        const userMessage = messages[messageIndex - 1];
        if (userMessage.role !== 'user') return;

        // Mark the assistant message as loading
        setMessages(prev => prev.map(msg => 
            msg.id === messageId ? { ...msg, isLoading: true, content: '' } : msg
        ));
        setIsLoading(true);

        try {
            const data = await queryLLM(userMessage.content, detailedMode);
            if (data) {
                setMessages(prev => prev.map(msg => 
                    msg.id === messageId 
                        ? {
                            ...msg,
                            content: data.message,
                            isLoading: false,
                            references: data.sources,
                            grounded: data.grounded,
                            embeddingsAreReal: data.embeddings_are_real,
                        }
                        : msg
                ));
            } else {
                setMessages(prev => prev.map(msg => 
                    msg.id === messageId 
                        ? { ...msg, content: 'Failed to regenerate response.', isLoading: false }
                        : msg
                ));
            }
        } catch (error) {
            setMessages(prev => prev.map(msg => 
                msg.id === messageId 
                    ? { ...msg, content: 'An error occurred while regenerating.', isLoading: false }
                    : msg
            ));
        } finally {
            setIsLoading(false);
        }
    };

    // Feedback handlers
    const handleFeedback = async (messageId: string, isPositive: boolean) => {
        try {
            await apiClient.post('/feedback', {
                message_id: messageId, 
                is_positive: isPositive
            });
            toast.success('Thanks for your feedback! This helps us improve.');
        } catch (error) {
            console.error('Feedback failed:', error);
            toast.error('Failed to submit feedback');
        }
    };

    // Settings handler
    const handleSettings = () => {
        setSettingsOpen(true);
    };

    // Attach handler
    const handleAttach = () => {
        navigate('/upload');
        toast('Please upload documents here to chat with them.', {
            icon: '📎',
        });
    };

    return (
        <Box sx={{ 
            display: 'flex', 
            flexDirection: 'column',
            height: '100vh',
            background: 'linear-gradient(180deg, #0F100E 0%, #0A0B09 50%, #171815 100%)',
        }}>
            {/* Header */}
            <Box sx={{ 
                p: 2, 
                borderBottom: '1px solid rgba(208, 162, 21, 0.08)',
                background: 'rgba(10, 15, 26, 0.8)',
                backdropFilter: 'blur(20px)',
            }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <IconButton onClick={() => navigate('/dashboard')} sx={{ color: 'text.secondary' }}>
                            <ArrowBack />
                        </IconButton>
                        <AutoAwesome sx={{ color: '#D0A215', fontSize: 28 }} />
                        <Box>
                            <Typography variant="h6" sx={{ fontWeight: 700, color: 'text.primary' }}>
                                Roneira <Box component="span" sx={{ color: '#D0A215' }}>Document Intelligence System</Box>
                            </Typography>
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                {documents.length} verified documents indexed
                            </Typography>
                        </Box>
                    </Box>
                    
                    <Box sx={{ display: 'flex', gap: 1 }}>
                        <Chip
                            label={detailedMode ? 'Detailed' : 'Precise'}
                            onClick={() => setDetailedMode(!detailedMode)}
                            sx={{
                                background: detailedMode 
                                    ? '#D0A215'
                                    : 'rgba(42, 43, 38, 1)',
                                color: 'white',
                                fontWeight: 600,
                                cursor: 'pointer',
                            }}
                        />
                        <Tooltip title="Settings">
                            <IconButton onClick={handleSettings} sx={{ color: 'text.secondary' }}>
                                <Settings />
                            </IconButton>
                        </Tooltip>
                    </Box>
                </Box>
            </Box>

            {/* Messages Area */}
            <Box sx={{ 
                flex: 1, 
                overflowY: 'auto', 
                p: 3,
                display: 'flex',
                flexDirection: 'column',
                gap: 3,
            }}>
                {messages.map((message) => (
                    <Box key={message.id} sx={{ 
                        display: 'flex',
                        justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start',
                    }}>
                        <Paper sx={{
                            maxWidth: '80%',
                            p: 2.5,
                            borderRadius: message.role === 'user' ? '20px 20px 4px 20px' : '20px 20px 20px 4px',
                            background: message.role === 'user'
                                ? '#D0A215'
                                : 'linear-gradient(135deg, rgba(10, 11, 9, 0.9) 0%, rgba(23, 24, 21, 0.7) 100%)',
                            border: message.role === 'assistant' ? '1px solid rgba(42, 43, 38, 1)' : 'none',
                        }}>
                            {message.isLoading ? (
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                    <CircularProgress size={20} sx={{ color: '#D0A215' }} />
                                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                                        Analyzing documents...
                                    </Typography>
                                </Box>
                            ) : (
                                <>
                                    <Typography 
                                        variant="body1" 
                                        sx={{ 
                                            color: 'text.primary',
                                            whiteSpace: 'pre-wrap',
                                            lineHeight: 1.7,
                                        }}
                                    >
                                        {message.content}
                                    </Typography>

                                    {/* Honesty signals. An answer the documents do not support, or
                                        one found by keyword matching rather than meaning, must not
                                        look identical to a fully grounded one. */}
                                    {message.role === 'assistant' && message.grounded !== undefined && (
                                        <Box sx={{ display: 'flex', gap: 1, mt: 1.5, flexWrap: 'wrap' }}>
                                            <Chip
                                                size="small"
                                                icon={message.grounded ? <VerifiedUser sx={{ fontSize: 14 }} /> : <WarningAmber sx={{ fontSize: 14 }} />}
                                                label={message.grounded ? 'Grounded in your documents' : 'Not supported by your documents'}
                                                sx={{
                                                    height: 22,
                                                    fontSize: '0.68rem',
                                                    color: message.grounded ? '#7A9A5B' : '#C9922B',
                                                    borderColor: message.grounded ? 'rgba(16,185,129,0.4)' : 'rgba(245,158,11,0.4)',
                                                    background: message.grounded ? 'rgba(16,185,129,0.1)' : 'rgba(245,158,11,0.1)',
                                                }}
                                                variant="outlined"
                                            />
                                            {message.embeddingsAreReal === false && (
                                                <Tooltip title="No embedding model is loaded, so search is matching keywords rather than meaning. Paraphrased questions may miss the right passage.">
                                                    <Chip
                                                        size="small"
                                                        icon={<WarningAmber sx={{ fontSize: 14 }} />}
                                                        label="Keyword-only search"
                                                        variant="outlined"
                                                        sx={{
                                                            height: 22,
                                                            fontSize: '0.68rem',
                                                            color: '#C9922B',
                                                            borderColor: 'rgba(245,158,11,0.4)',
                                                            background: 'rgba(245,158,11,0.1)',
                                                        }}
                                                    />
                                                </Tooltip>
                                            )}
                                        </Box>
                                    )}

                                    {/* Document References */}
                                    {message.references && message.references.length > 0 && (
                                        <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid rgba(42, 43, 38, 1)' }}>
                                            <Typography variant="caption" sx={{ color: '#D0A215', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
                                                <Description sx={{ fontSize: 16 }} />
                                                Sources ({message.references.length})
                                            </Typography>
                                            
                                            <Box sx={{ mt: 1, display: 'flex', flexDirection: 'column', gap: 1 }}>
                                                {message.references.map((ref, idx) => (
                                                    <Box
                                                        key={ref.chunk_id}
                                                        sx={{
                                                            p: 1.5,
                                                            borderRadius: 2,
                                                            background: 'rgba(208, 162, 21, 0.08)',
                                                            border: '1px solid rgba(42, 43, 38, 1)',
                                                            cursor: 'pointer',
                                                        }}
                                                    >
                                                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                                            <Box
                                                                sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: 1 }}
                                                                onClick={() => toggleRefExpand(ref.chunk_id)}
                                                            >
                                                                <Chip
                                                                    label={idx + 1}
                                                                    size="small"
                                                                    sx={{
                                                                        width: 24,
                                                                        height: 24,
                                                                        fontSize: '0.7rem',
                                                                        background: '#D0A215'
                                                                    }}
                                                                />
                                                                <Typography variant="body2" sx={{ fontWeight: 500, color: 'text.primary' }}>
                                                                    {ref.filename ?? 'Unknown document'}
                                                                </Typography>
                                                                {/* The page is what makes a citation checkable. */}
                                                                {ref.page_number !== null && ref.page_number !== undefined && (
                                                                    <Chip
                                                                        label={`p. ${ref.page_number}`}
                                                                        size="small"
                                                                        variant="outlined"
                                                                        sx={{ height: 20, fontSize: '0.65rem', color: '#D0A215', borderColor: 'rgba(6,182,212,0.4)' }}
                                                                    />
                                                                )}
                                                                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                                                    {Math.round(ref.score * 100)}% match
                                                                </Typography>
                                                            </Box>
                                                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                                                <Tooltip title="Download Document">
                                                                    <IconButton 
                                                                        size="small" 
                                                                        onClick={(e) => {
                                                                            e.stopPropagation();
                                                                            handleDownload(ref.document_id, ref.filename ?? ref.document_id);
                                                                        }}
                                                                        sx={{ color: '#D0A215', mr: 1 }}
                                                                    >
                                                                        <Download fontSize="small" />
                                                                    </IconButton>
                                                                </Tooltip>
                                                                <IconButton size="small" onClick={() => toggleRefExpand(ref.chunk_id)}>
                                                                    {expandedRefs.includes(ref.chunk_id) ? <ExpandLess /> : <ExpandMore />}
                                                                </IconButton>
                                                            </Box>
                                                        </Box>
                                                        
                                                        <Collapse in={expandedRefs.includes(ref.chunk_id)}>
                                                            <Typography 
                                                                variant="caption" 
                                                                sx={{ 
                                                                    display: 'block',
                                                                    mt: 1,
                                                                    color: 'text.secondary',
                                                                    background: 'rgba(0,0,0,0.2)',
                                                                    p: 1,
                                                                    borderRadius: 1,
                                                                    fontFamily: 'monospace',
                                                                    fontSize: '0.7rem',
                                                                }}
                                                            >
                                                                {ref.content_preview}
                                                            </Typography>
                                                        </Collapse>
                                                    </Box>
                                                ))}
                                            </Box>
                                        </Box>
                                    )}
                                    
                                    {/* Action Buttons */}
                                    {message.role === 'assistant' && !message.isLoading && message.id !== 'welcome' && (
                                        <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                                            <Tooltip title="Copy">
                                                <IconButton size="small" onClick={() => copyToClipboard(message.content)}>
                                                    <ContentCopy sx={{ fontSize: 16, color: 'text.secondary' }} />
                                                </IconButton>
                                            </Tooltip>
                                            <Tooltip title="Regenerate">
                                                <IconButton size="small" onClick={() => regenerateResponse(message.id)}>
                                                    <Refresh sx={{ fontSize: 16, color: 'text.secondary' }} />
                                                </IconButton>
                                            </Tooltip>
                                            <Tooltip title="Good response">
                                                <IconButton size="small" onClick={() => handleFeedback(message.id, true)}>
                                                    <ThumbUp sx={{ fontSize: 16, color: 'text.secondary' }} />
                                                </IconButton>
                                            </Tooltip>
                                            <Tooltip title="Poor response">
                                                <IconButton size="small" onClick={() => handleFeedback(message.id, false)}>
                                                    <ThumbDown sx={{ fontSize: 16, color: 'text.secondary' }} />
                                                </IconButton>
                                            </Tooltip>
                                        </Box>
                                    )}
                                </>
                            )}
                        </Paper>
                    </Box>
                ))}
                
                <div ref={messagesEndRef} />
            </Box>

            {/* Input Area */}
            <Box sx={{ 
                p: 3,
                borderTop: '1px solid rgba(208, 162, 21, 0.08)',
                background: 'rgba(10, 15, 26, 0.8)',
                backdropFilter: 'blur(20px)',
            }}>
                <Paper sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    p: 1,
                    pl: 2,
                    borderRadius: 4,
                    background: 'rgba(10, 11, 9, 0.8)',
                    border: '2px solid rgba(42, 43, 38, 1)',
                    '&:focus-within': {
                        border: '2px solid rgba(208, 162, 21, 0.5)',
                        boxShadow: '0 0 20px rgba(42, 43, 38, 1)',
                    }
                }}>
                    <TextField
                        ref={inputRef}
                        fullWidth
                        placeholder="Ask about your documents..."
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
                        multiline
                        maxRows={4}
                        variant="standard"
                        InputProps={{ disableUnderline: true }}
                        sx={{
                            '& .MuiInputBase-input': {
                                color: 'text.primary',
                                fontSize: '1rem',
                            }
                        }}
                    />
                    
                    <Tooltip title="Attach document">
                        <IconButton onClick={handleAttach} sx={{ color: 'text.secondary' }}>
                            <AttachFile />
                        </IconButton>
                    </Tooltip>
                    
                    <IconButton 
                        onClick={handleSend}
                        disabled={!inputValue.trim() || isLoading}
                        sx={{ 
                            background: '#D0A215',
                            color: 'white',
                            '&:hover': {
                                background: '#B58E12',
                            },
                            '&:disabled': {
                                background: 'rgba(42, 43, 38, 1)',
                                color: 'rgba(255,255,255,0.5)',
                            }
                        }}
                    >
                        <Send />
                    </IconButton>
                </Paper>
            </Box>

            {/* Settings Drawer */}
            <Drawer
                anchor="right"
                open={settingsOpen}
                onClose={() => setSettingsOpen(false)}
                PaperProps={{
                    sx: {
                        width: 320,
                        background: 'linear-gradient(180deg, #0A0B09 0%, #171815 100%)',
                        borderLeft: '1px solid rgba(42, 43, 38, 1)',
                        p: 3
                    }
                }}
            >
                <Typography variant="h6" sx={{ fontWeight: 700, color: 'white', mb: 3 }}>
                    Chat Settings
                </Typography>
                
                <Divider sx={{ borderColor: 'rgba(255,255,255,0.1)', mb: 3 }} />
                
                {/* RAG Toggle */}
                <FormControlLabel
                    control={
                        <Switch 
                            checked={ragEnabled} 
                            onChange={(e) => setRagEnabled(e.target.checked)}
                            sx={{ '& .MuiSwitch-thumb': { bgcolor: '#D0A215' } }}
                        />
                    }
                    label={<Typography sx={{ color: 'text.primary' }}>Enable RAG</Typography>}
                    sx={{ mb: 2 }}
                />
                <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 3 }}>
                    Use document context for answers
                </Typography>
                
                {/* Temperature Slider */}
                <Typography sx={{ color: 'text.primary', mb: 1 }}>Temperature: {temperature}</Typography>
                <Slider
                    value={temperature}
                    onChange={(_, val) => setTemperature(val as number)}
                    min={0}
                    max={1}
                    step={0.1}
                    sx={{ color: '#D0A215', mb: 3 }}
                />
                <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 3 }}>
                    Lower = more focused, Higher = more creative
                </Typography>
                
                {/* Max Tokens Slider */}
                <Typography sx={{ color: 'text.primary', mb: 1 }}>Max Tokens: {maxTokens}</Typography>
                <Slider
                    value={maxTokens}
                    onChange={(_, val) => setMaxTokens(val as number)}
                    min={128}
                    max={2048}
                    step={128}
                    sx={{ color: '#D0A215', mb: 3 }}
                />
                <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 3 }}>
                    Maximum response length
                </Typography>
                
                <Divider sx={{ borderColor: 'rgba(255,255,255,0.1)', mb: 3 }} />
                
                {/* Model Info */}
                <Typography variant="body2" sx={{ color: 'text.secondary', mb: 1 }}>Current Model</Typography>
                <Chip 
                    label="llama3.2:3b" 
                    size="small" 
                    sx={{ bgcolor: 'rgba(42, 43, 38, 1)', color: '#D0A215', mb: 3 }}
                />
                
                <Typography variant="body2" sx={{ color: 'text.secondary', mb: 1 }}>Session ID</Typography>
                <Typography variant="caption" sx={{ color: '#D0A215', fontFamily: 'monospace', wordBreak: 'break-all' }}>
                    {sessionId}
                </Typography>
                
                <Box sx={{ mt: 'auto', pt: 4 }}>
                    <Button 
                        fullWidth 
                        variant="outlined" 
                        onClick={() => setSettingsOpen(false)}
                        sx={{ borderColor: 'rgba(42, 43, 38, 1)', color: 'text.primary' }}
                    >
                        Close
                    </Button>
                </Box>
            </Drawer>
        </Box>
    );
};

export default AIChat;
