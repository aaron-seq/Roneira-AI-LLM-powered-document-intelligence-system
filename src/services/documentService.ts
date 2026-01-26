import { apiClient } from '../api/client';

export interface Document {
    id: string;
    filename: string;
    status: 'queued' | 'processing' | 'completed' | 'failed';
    upload_timestamp: string;
}

export const DocumentService = {
    // Fetch all documents
    async listDocuments(limit = 10, offset = 0): Promise<Document[]> {
        const response = await apiClient.get('/documents', {
            params: { limit, offset },
        });
        return response.data.documents;
    },

    // Upload a document
    async uploadDocument(file: File): Promise<Document> {
        const formData = new FormData();
        formData.append('file', file);

        const response = await apiClient.post('/documents/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },

    // Get status
    async getStatus(documentId: string): Promise<Document> {
        const response = await apiClient.get(`/documents/${documentId}/status`);
        return response.data;
    }
};
