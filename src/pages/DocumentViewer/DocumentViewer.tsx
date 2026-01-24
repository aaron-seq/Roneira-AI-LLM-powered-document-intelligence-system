import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
    Box,
    Typography,
    Paper,
    TextField,
    Button,
    CircularProgress,
    Grid,
} from '@mui/material';
import axios from 'axios';
import toast from 'react-hot-toast';

// API path handled by Vite proxy
// const API_BASE = 'http://localhost:8000';

const DocumentViewer = () => {
    const { documentId } = useParams();
    const [documentData, setDocumentData] = useState<any>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [question, setQuestion] = useState('');
    const [chatHistory, setChatHistory] = useState<any[]>([]);
    const [isAnswering, setIsAnswering] = useState(false);

    useEffect(() => {
        const fetchDocument = async () => {
            try {
                // Fetch actual document data from the backend
                // The status endpoint returns the processing result which includes the text
                const response = await axios.get(`/api/documents/${documentId}/status`);
                setDocumentData(response.data);
            } catch (error) {
                console.error('Failed to load document:', error);
                toast.error('Failed to load document data.');
            } finally {
                setIsLoading(false);
            }
        };
        if (documentId) {
            fetchDocument();
        }
    }, [documentId]);

    const handleAskQuestion = async () => {
        if (!question.trim()) return;
        setIsAnswering(true);

        const newHistory = [...chatHistory, { role: 'user', content: question }];
        setChatHistory(newHistory);

        try {
            // Call the real /api/chat endpoint for document-based Q&A
            const response = await axios.post('/api/chat', {
                message: question,
                session_id: 'doc-chat-session', 
                use_rag: true,
                rag_top_k: 3,
                document_id: documentId, // Filter by this document
                detailed: false
            });

            setChatHistory([...newHistory, { role: 'assistant', content: response.data.message }]);
            setQuestion('');

        } catch (error) {
            toast.error('Failed to get an answer.');
            setChatHistory(newHistory); // Revert history on error
        } finally {
            setIsAnswering(false);
        }
    };

    if (isLoading) {
        return <CircularProgress />;
    }

    return (
        <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
                <Typography variant="h5" gutterBottom>
                    Document Content
                </Typography>
                <Paper sx={{ p: 2, height: '70vh', overflowY: 'auto' }}>
                    <Typography variant="h6">Extracted Text</Typography>
                    <Typography paragraph sx={{ whiteSpace: 'pre-wrap' }}>
                        {documentData?.result?.original_text || documentData?.text || 'No text extracted yet.'}
                    </Typography>
                    {(documentData?.result?.ai_analysis || documentData?.analysis) && (
                        <>
                            <Typography variant="h6" sx={{ mt: 2 }}>Analysis</Typography>
                            <Typography paragraph>
                                {typeof (documentData.result?.ai_analysis || documentData.analysis) === 'string' 
                                    ? (documentData.result?.ai_analysis || documentData.analysis)
                                    : JSON.stringify(documentData.result?.ai_analysis || documentData.analysis, null, 2)}
                            </Typography>
                        </>
                    )}
                </Paper>
            </Grid>
            <Grid item xs={12} md={6}>
                <Typography variant="h5" gutterBottom>
                    Chat with Document
                </Typography>
                <Paper sx={{ p: 2, height: '70vh', display: 'flex', flexDirection: 'column' }}>
                    <Box sx={{ flexGrow: 1, overflowY: 'auto', mb: 2 }}>
                        {chatHistory.map((entry, index) => (
                            <Box key={index} sx={{ mb: 1, textAlign: entry.role === 'user' ? 'right' : 'left' }}>
                                <Typography variant="caption" color="text.secondary">{entry.role}</Typography>
                                <Paper sx={{ p: 1, display: 'inline-block', bgcolor: entry.role === 'user' ? 'primary.light' : 'grey.200' }}>
                                    {entry.content}
                                </Paper>
                            </Box>
                        ))}
                    </Box>
                    <Box sx={{ display: 'flex' }}>
                        <TextField
                            fullWidth
                            variant="outlined"
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
        </Grid>
    );
};

export default DocumentViewer;
