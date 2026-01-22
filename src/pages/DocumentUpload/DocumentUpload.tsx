import { useCallback, useState, useEffect, useRef } from 'react';
import { useDropzone, FileWithPath } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import {
    Box,
    Button,
    Typography,
    Paper,
    CircularProgress,
    LinearProgress,
    List,
    ListItem,
    ListItemText,
    ListItemIcon,
    IconButton,
} from '@mui/material';
import { UploadFile as UploadFileIcon, CheckCircle, Error as ErrorIcon, Delete as DeleteIcon } from '@mui/icons-material';
import axios from 'axios';
import toast from 'react-hot-toast';

// Define a type for the processing status message from the backend
interface ProcessingStatus {
    document_id: string;
    status: 'pending' | 'queued' | 'processing' | 'analyzing' | 'enhancing' | 'completed' | 'failed';
    progress: number;
    message?: string;
}

// Store the actual File object reference, not spread properties
interface FileWithStatus {
    file: File; // The actual File object
    id?: string; // document_id after upload
    uploadStatus: 'idle' | 'uploading' | 'uploaded' | 'error';
    uploadProgress: number;
}

const DocumentUpload = () => {
    const navigate = useNavigate();
    const [files, setFiles] = useState<FileWithStatus[]>([]);
    const [processingStatuses, setProcessingStatuses] = useState<Record<string, ProcessingStatus>>({});
    const [isUploading, setIsUploading] = useState(false);
    
    // Track active polling intervals to clear them later
    const activePolls = useRef<Record<string, ReturnType<typeof setInterval>>>({});

    // Polling function for a single document
    const pollStatus = useCallback(async (docId: string) => {
        try {
            const response = await axios.get(`/api/documents/${docId}/status`);
            const status = response.data;
            
            if (status.document_id === docId) {
                setProcessingStatuses(prev => ({ ...prev, [docId]: status }));
                
                if (status.status === 'completed') {
                    toast.success(`Document processing completed!`);
                    return true; // Stop polling
                } else if (status.status === 'failed') {
                    toast.error(`Document processing failed`);
                    return true; // Stop polling
                }
            }
        } catch (error) {
            console.error(`Polling error for ${docId}:`, error);
        }
        return false;
    }, []);

    // Start polling for a specific document
    const startPollingForDoc = useCallback((docId: string) => {
        if (activePolls.current[docId]) return; // Already polling

        console.log(`Starting polling for ${docId}...`);
        activePolls.current[docId] = setInterval(async () => {
            const shouldStop = await pollStatus(docId);
            if (shouldStop) {
                clearInterval(activePolls.current[docId]);
                delete activePolls.current[docId];
            }
        }, 2000);
    }, [pollStatus]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            Object.values(activePolls.current).forEach(clearInterval);
        };
    }, []);

    const onDrop = useCallback((acceptedFiles: FileWithPath[]) => {
        const newFiles: FileWithStatus[] = acceptedFiles.map(file => ({
            file: file as File, // Store the actual File reference
            uploadStatus: 'idle',
            uploadProgress: 0
        }));
        setFiles(prev => [...prev, ...newFiles]);
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'application/pdf': ['.pdf'] },
        multiple: true,
    });

    const removeFile = (index: number) => {
        setFiles(prev => prev.filter((_, i) => i !== index));
    };

    const handleUpload = async () => {
        if (files.length === 0) return;

        setIsUploading(true);
        const filesToUpload = files.filter(f => f.uploadStatus === 'idle' || f.uploadStatus === 'error');

        await Promise.all(filesToUpload.map(async (fileWrapper, idx) => {
            // Find current index to update state
            const fileIndex = files.indexOf(fileWrapper);
            const updateFileState = (updates: Partial<FileWithStatus>) => {
                 setFiles(prev => prev.map((f, i) => 
                    i === fileIndex ? { ...f, ...updates } : f
                 ));
            };

            updateFileState({ uploadStatus: 'uploading', uploadProgress: 0 });

            const formData = new FormData();
            formData.append('file', fileWrapper.file); // Use the stored File reference

            try {
                const response = await axios.post('/api/documents/upload', formData, {
                    onUploadProgress: (progressEvent) => {
                        if (progressEvent.total) {
                            const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                            updateFileState({ uploadProgress: percent });
                        }
                    },
                    headers: { 'Content-Type': 'multipart/form-data' },
                });

                const { document_id } = response.data;
                updateFileState({ uploadStatus: 'uploaded', id: document_id, uploadProgress: 100 });
                
                toast.success(`Uploaded ${fileWrapper.file.name}`);
                
                // Initialize status tracking
                setProcessingStatuses(prev => ({
                    ...prev, 
                    [document_id]: { document_id, status: 'queued', progress: 0 }
                }));

                // Start polling immediately
                startPollingForDoc(document_id);

            } catch (error: any) {
                console.error(`Upload failed for ${fileWrapper.file.name}`, error);
                updateFileState({ uploadStatus: 'error' });
                const errorMessage = error.response?.data?.detail || error.message || 'Unknown error';
                toast.error(`Failed to upload ${fileWrapper.file.name}: ${errorMessage}`);
            }
        }));

        setIsUploading(false);
    };

    const allCompleted = files.length > 0 && files.every(f => {
        if (f.uploadStatus !== 'uploaded') return false;
        const status = processingStatuses[f.id!];
        return status?.status === 'completed';
    });

    return (
        <Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h4">Upload Documents</Typography>
                {allCompleted && (
                    <Button variant="outlined" onClick={() => navigate('/dashboard')}>
                        Go to Dashboard
                    </Button>
                )}
            </Box>

            <Paper
                {...getRootProps()}
                sx={{
                    p: 4,
                    textAlign: 'center',
                    border: `2px dashed ${isDragActive ? 'primary.main' : 'grey.500'}`,
                    cursor: isUploading ? 'not-allowed' : 'pointer',
                    bgcolor: isDragActive ? 'action.hover' : 'background.paper',
                    mb: 3,
                    opacity: isUploading ? 0.7 : 1
                }}
            >
                <input {...getInputProps()} disabled={isUploading} />
                <UploadFileIcon sx={{ fontSize: 60, color: 'grey.500' }} />
                <Typography>
                    {isDragActive ? 'Drop files here ...' : 'Drag & drop PDFs here, or click to select multiple files'}
                </Typography>
            </Paper>

            {files.length > 0 && (
                <Paper sx={{ mb: 3, p: 2 }}>
                     <List>
                        {files.map((fileWrapper, index) => {
                            const docId = fileWrapper.id;
                            const procStatus = docId ? processingStatuses[docId] : null;
                            const isError = fileWrapper.uploadStatus === 'error' || procStatus?.status === 'failed';
                            const isDone = procStatus?.status === 'completed';

                            return (
                                <ListItem key={index} divider>
                                    <ListItemIcon>
                                        {isDone ? <CheckCircle color="success" /> : 
                                         isError ? <ErrorIcon color="error" /> : 
                                         <UploadFileIcon />}
                                    </ListItemIcon>
                                    <ListItemText 
                                        primary={fileWrapper.file.name}
                                        secondary={
                                            <Box sx={{ width: '100%', mt: 1 }}>
                                                {fileWrapper.uploadStatus === 'uploading' && (
                                                    <>
                                                        <Typography variant="caption">Uploading... {fileWrapper.uploadProgress}%</Typography>
                                                        <LinearProgress variant="determinate" value={fileWrapper.uploadProgress} />
                                                    </>
                                                )}
                                                {fileWrapper.uploadStatus === 'uploaded' && procStatus && (
                                                    <>
                                                        <Typography variant="caption">
                                                            {procStatus.status.toUpperCase()} - {procStatus.message || `${procStatus.progress}%`}
                                                        </Typography>
                                                        <LinearProgress 
                                                            variant={procStatus.status === 'queued' ? 'indeterminate' : 'determinate'} 
                                                            value={procStatus.progress} 
                                                            color={isDone ? "success" : "primary"}
                                                        />
                                                    </>
                                                )}
                                                {fileWrapper.uploadStatus === 'idle' && (
                                                    <Typography variant="caption" color="text.secondary">Ready to upload</Typography>
                                                )}
                                                {fileWrapper.uploadStatus === 'error' && (
                                                    <Typography variant="caption" color="error">Upload Failed</Typography>
                                                )}
                                            </Box>
                                        }
                                    />
                                    {!isUploading && fileWrapper.uploadStatus === 'idle' && (
                                        <IconButton onClick={() => removeFile(index)}>
                                            <DeleteIcon />
                                        </IconButton>
                                    )}
                                    {isDone && (
                                         <Button size="small" onClick={() => navigate(`/documents/${docId}`)}>
                                            View
                                         </Button>
                                    )}
                                </ListItem>
                            );
                        })}
                     </List>
                     
                     <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
                        <Button onClick={() => setFiles([])} disabled={isUploading}>Clear All</Button>
                        <Button 
                            variant="contained" 
                            onClick={handleUpload} 
                            disabled={isUploading || files.filter(f => f.uploadStatus === 'idle').length === 0}
                            startIcon={isUploading ? <CircularProgress size={20} /> : null}
                        >
                            {isUploading ? 'Processing...' : `Upload ${files.filter(f => f.uploadStatus === 'idle').length} Files`}
                        </Button>
                     </Box>
                </Paper>
            )}
        </Box>
    );
};

export default DocumentUpload;
