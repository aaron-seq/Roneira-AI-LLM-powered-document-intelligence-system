import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Paper,
  Typography,
  LinearProgress,
  Chip,
  Stack,
  alpha,
} from '@mui/material';
import {
  CloudUpload,
  Description,
  CheckCircle,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';

import { useDocumentUpload } from '@/hooks/useDocumentUpload';
import { formatFileSize } from '@/utils/fileUtils';

interface DocumentUploadZoneProps {
  onUploadSuccess?: (documentId: string) => void;
  onUploadError?: (error: string) => void;
}

export const DocumentUploadZone: React.FC<DocumentUploadZoneProps> = ({
  onUploadSuccess,
  onUploadError,
}) => {
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const { uploadDocument, isLoading } = useDocumentUpload();

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;

      try {
        setUploadProgress(0);
        
        const result = await uploadDocument(file, {
          onUploadProgress: (progress) => {
            setUploadProgress(progress);
          },
        });

        onUploadSuccess?.(result.document_id);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Upload failed';
        onUploadError?.(errorMessage);
      }
    },
    [uploadDocument, onUploadSuccess, onUploadError]
  );

  const {
    getRootProps,
    getInputProps,
    isDragActive,
    isDragReject,
    acceptedFiles,
    fileRejections,
  } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
    },
    maxSize: 50 * 1024 * 1024, // 50MB
    multiple: false,
  });

  const getDropzoneStyles = () => {
    if (isDragReject) return { borderColor: 'error.main', backgroundColor: alpha('#f44336', 0.04) };
    if (isDragActive) return { borderColor: 'primary.main', backgroundColor: alpha('#1976d2', 0.04) };
    return { borderColor: 'grey.300', backgroundColor: 'background.paper' };
  };

  return (
    <Paper elevation={2} sx={{ p: 3 }}>
      <Box
        {...getRootProps()}
        sx={{
          border: '2px dashed',
          borderRadius: 2,
          p: 6,
          textAlign: 'center',
          cursor: 'pointer',
          transition: 'all 0.2s ease-in-out',
          minHeight: 200,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          ...getDropzoneStyles(),
          '&:hover': {
            backgroundColor: alpha('#1976d2', 0.02),
            borderColor: 'primary.light',
          },
        }}
      >
        <input {...getInputProps()} />
        
        <AnimatePresence mode="wait">
          {isLoading ? (
            <motion.div
              key="uploading"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
            >
              <CloudUpload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                Uploading Document...
              </Typography>
              <Box sx={{ width: '100%', maxWidth: 300, mt: 2 }}>
                <LinearProgress
                  variant="determinate"
                  value={uploadProgress}
                  sx={{ height: 8, borderRadius: 4 }}
                />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  {uploadProgress}% complete
                </Typography>
              </Box>
            </motion.div>
          ) : isDragActive ? (
            <motion.div
              key="drag-active"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <CloudUpload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
              <Typography variant="h6" color="primary">
                Drop your document here!
              </Typography>
            </motion.div>
          ) : (
            <motion.div
              key="default"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <Description sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                Drag & drop your document here
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                or click to browse files
              </Typography>
              
              <Stack direction="row" spacing={1} justifyContent="center" flexWrap="wrap">
                {['PDF', 'DOCX', 'DOC', 'TXT', 'PNG', 'JPG'].map((format) => (
                  <Chip
                    key={format}
                    label={format}
                    size="small"
                    variant="outlined"
                    sx={{ fontSize: '0.75rem' }}
                  />
                ))}
              </Stack>
              
              <Typography variant="caption" display="block" sx={{ mt: 1, color: 'text.secondary' }}>
                Max file size: 50MB
              </Typography>
            </motion.div>
          )}
        </AnimatePresence>
      </Box>

      {/* File Rejections */}
      {fileRejections.length > 0 && (
        <Box sx={{ mt: 2 }}>
          {fileRejections.map(({ file, errors }) => (
            <Box
              key={file.name}
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                p: 2,
                backgroundColor: alpha('#f44336', 0.04),
                border: '1px solid',
                borderColor: 'error.main',
                borderRadius: 1,
                mb: 1,
              }}
            >
              <ErrorIcon color="error" />
              <Box>
                <Typography variant="body2" fontWeight="medium">
                  {file.name} - {formatFileSize(file.size)}
                </Typography>
                {errors.map((error) => (
                  <Typography key={error.code} variant="caption" color="error">
                    {error.message}
                  </Typography>
                ))}
              </Box>
            </Box>
          ))}
        </Box>
      )}

      {/* Accepted Files */}
      {acceptedFiles.length > 0 && !isLoading && (
        <Box sx={{ mt: 2 }}>
          {acceptedFiles.map((file) => (
            <Box
              key={file.name}
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                p: 2,
                backgroundColor: alpha('#4caf50', 0.04),
                border: '1px solid',
                borderColor: 'success.main',
                borderRadius: 1,
              }}
            >
              <CheckCircle color="success" />
              <Typography variant="body2">
                {file.name} - {formatFileSize(file.size)}
              </Typography>
            </Box>
          ))}
        </Box>
      )}
    </Paper>
  );
};
