// ==============================================================================
// API Configuration for Free Deployment
// ==============================================================================

// Get API base URL from environment or default to current host
const getApiBaseUrl = (): string => {
  // Check if we're in development
  if (import.meta.env.DEV) {
    return import.meta.env.VITE_API_URL || 'http://localhost:8000';
  }
  
  // Production: use same origin or specified URL
  return import.meta.env.VITE_API_URL || window.location.origin;
};

export const API_CONFIG = {
  BASE_URL: getApiBaseUrl(),
  TIMEOUT: 30000, // 30 seconds
  MAX_FILE_SIZE: 50 * 1024 * 1024, // 50MB
  ALLOWED_FILE_TYPES: [
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'image/jpeg',
    'image/jpg', 
    'image/png',
    'image/webp',
    'text/plain'
  ],
  POLL_INTERVAL: 2000, // 2 seconds
  MAX_RETRIES: 3
};

export const API_ENDPOINTS = {
  HEALTH: '/health',
  UPLOAD: '/upload',
  DOCUMENTS: '/documents',
  DOCUMENT_STATUS: (id: string) => `/documents/${id}/status`,
  DOCUMENT_ANALYSIS: (id: string) => `/documents/${id}`,
};

// API Response Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  file_size: number;
  status: string;
  message: string;
}

export interface ProcessingStatus {
  document_id: string;
  status: 'uploaded' | 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  error?: string;
}

export interface DocumentAnalysis {
  document_id: string;
  text: string;
  analysis: {
    analysis?: any;
    insights?: any;
    ocr_details?: any;
  };
  confidence: number;
  processing_time: number;
  status: string;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  services: {
    llm: { status: string; services_available?: string[] };
    ocr: { status: string; services_available?: string[] };
    redis: { status: string; error?: string };
    database: { status: string; error?: string };
  };
  timestamp: string;
}