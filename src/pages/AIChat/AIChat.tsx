import React, { useState, useRef, useEffect } from 'react';
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
    Button
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
    Download
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

// ==============================================================================
// Roneira AI - AI Chat Interface
// Document Intelligence with RAG (Retrieval Augmented Generation)
// ==============================================================================

const API_BASE = 'http://localhost:8000';

interface DocumentReference {
    id: string;
    filename: string;
    relevance: number;
    excerpt: string;
}

interface ChatMessage {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    references?: DocumentReference[];
    isLoading?: boolean;
    isDetailed?: boolean;
}

const AIChat = () => {
    const navigate = useNavigate();
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [documents, setDocuments] = useState<any[]>([]);
    const [detailedMode, setDetailedMode] = useState(false);
    const [expandedRefs, setExpandedRefs] = useState<string[]>([]);
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
            const response = await fetch(`${API_BASE}/documents?limit=100`);
            if (response.ok) {
                const data = await response.json();
                setDocuments(data);
            }
        } catch (error) {
            console.error('Failed to fetch documents:', error);
        }
    };

    // Download Document
    const handleDownload = (docId: string, filename: string) => {
        window.open(`${API_BASE}/documents/${docId}/download`, '_blank');
    };

    // Search documents for relevant context
    const searchDocuments = async (query: string): Promise<DocumentReference[]> => {
        // We now rely on the backend query endpoint to return sources, 
        // but for immediate UI feedback or fallback, we can keep client-side logic if needed.
        // However, the main RAG logic is now server-side in /query.
        return []; 
    };

    // Query LLM with document context
    const queryLLM = async (query: string, isDetailed: boolean) => {
        try {
            const response = await fetch(`${API_BASE}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    query: query,
                    max_tokens: isDetailed ? 2048 : 512,
                    detailed: isDetailed
                })
            });

            if (response.ok) {
                return await response.json();
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
                            content: data.response, 
                            isLoading: false, 
                            references: data.sources 
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
    };

    return (
        <Box sx={{ 
            display: 'flex', 
            flexDirection: 'column',
            height: '100vh',
            background: 'linear-gradient(180deg, #0a0f1a 0%, #0f172a 50%, #1e293b 100%)',
        }}>
            {/* Header */}
            <Box sx={{ 
                p: 2, 
                borderBottom: '1px solid rgba(99, 102, 241, 0.1)',
                background: 'rgba(10, 15, 26, 0.8)',
                backdropFilter: 'blur(20px)',
            }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <IconButton onClick={() => navigate('/dashboard')} sx={{ color: 'text.secondary' }}>
                            <ArrowBack />
                        </IconButton>
                        <AutoAwesome sx={{ color: '#6366f1', fontSize: 28 }} />
                        <Box>
                            <Typography variant="h6" sx={{ fontWeight: 700, color: 'text.primary' }}>
                                Roneira <Box component="span" sx={{ color: '#06b6d4' }}>Document Intelligence System</Box>
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
                                    ? 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)'
                                    : 'rgba(99, 102, 241, 0.2)',
                                color: 'white',
                                fontWeight: 600,
                                cursor: 'pointer',
                            }}
                        />
                        <Tooltip title="Settings">
                            <IconButton sx={{ color: 'text.secondary' }}>
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
                                ? 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)'
                                : 'linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%)',
                            border: message.role === 'assistant' ? '1px solid rgba(99, 102, 241, 0.2)' : 'none',
                        }}>
                            {message.isLoading ? (
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                    <CircularProgress size={20} sx={{ color: '#06b6d4' }} />
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
                                    
                                    {/* Document References */}
                                    {message.references && message.references.length > 0 && (
                                        <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid rgba(99, 102, 241, 0.2)' }}>
                                            <Typography variant="caption" sx={{ color: '#06b6d4', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
                                                <Description sx={{ fontSize: 16 }} />
                                                Sources ({message.references.length})
                                            </Typography>
                                            
                                            <Box sx={{ mt: 1, display: 'flex', flexDirection: 'column', gap: 1 }}>
                                                {message.references.map((ref, idx) => (
                                                    <Box 
                                                        key={ref.id}
                                                        sx={{
                                                            p: 1.5,
                                                            borderRadius: 2,
                                                            background: 'rgba(99, 102, 241, 0.1)',
                                                            border: '1px solid rgba(99, 102, 241, 0.15)',
                                                            cursor: 'pointer',
                                                        }}
                                                    >
                                                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                                            <Box 
                                                                sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: 1 }}
                                                                onClick={() => toggleRefExpand(ref.id)}
                                                            >
                                                                <Chip 
                                                                    label={idx + 1} 
                                                                    size="small" 
                                                                    sx={{ 
                                                                        width: 24, 
                                                                        height: 24,
                                                                        fontSize: '0.7rem',
                                                                        background: '#6366f1' 
                                                                    }} 
                                                                />
                                                                <Typography variant="body2" sx={{ fontWeight: 500, color: 'text.primary' }}>
                                                                    {ref.filename}
                                                                </Typography>
                                                            </Box>
                                                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                                                <Tooltip title="Download Document">
                                                                    <IconButton 
                                                                        size="small" 
                                                                        onClick={(e) => {
                                                                            e.stopPropagation();
                                                                            handleDownload(ref.id, ref.filename);
                                                                        }}
                                                                        sx={{ color: '#06b6d4', mr: 1 }}
                                                                    >
                                                                        <Download fontSize="small" />
                                                                    </IconButton>
                                                                </Tooltip>
                                                                <IconButton size="small" onClick={() => toggleRefExpand(ref.id)}>
                                                                    {expandedRefs.includes(ref.id) ? <ExpandLess /> : <ExpandMore />}
                                                                </IconButton>
                                                            </Box>
                                                        </Box>
                                                        
                                                        <Collapse in={expandedRefs.includes(ref.id)}>
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
                                                                {ref.excerpt}
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
                                            <Tooltip title="Good response">
                                                <IconButton size="small">
                                                    <ThumbUp sx={{ fontSize: 16, color: 'text.secondary' }} />
                                                </IconButton>
                                            </Tooltip>
                                            <Tooltip title="Poor response">
                                                <IconButton size="small">
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
                borderTop: '1px solid rgba(99, 102, 241, 0.1)',
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
                    background: 'rgba(15, 23, 42, 0.8)',
                    border: '2px solid rgba(99, 102, 241, 0.2)',
                    '&:focus-within': {
                        border: '2px solid rgba(99, 102, 241, 0.5)',
                        boxShadow: '0 0 20px rgba(99, 102, 241, 0.2)',
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
                        <IconButton sx={{ color: 'text.secondary' }}>
                            <AttachFile />
                        </IconButton>
                    </Tooltip>
                    
                    <IconButton 
                        onClick={handleSend}
                        disabled={!inputValue.trim() || isLoading}
                        sx={{ 
                            background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                            color: 'white',
                            '&:hover': {
                                background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
                            },
                            '&:disabled': {
                                background: 'rgba(99, 102, 241, 0.3)',
                                color: 'rgba(255,255,255,0.5)',
                            }
                        }}
                    >
                        <Send />
                    </IconButton>
                </Paper>
            </Box>
        </Box>
    );
};

export default AIChat;
