import axios, { AxiosError } from 'axios';
import { v4 as uuidv4 } from 'uuid';

// Environment variable input
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

/**
 * Custom API Error for consistent frontend error handling.
 */
export class ApiError extends Error {
    constructor(
        public message: string,
        public statusCode?: number,
        public details?: any
    ) {
        super(message);
        this.name = 'ApiError';
    }
}

export const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
    timeout: 30000, // 30s timeout
});

// Interceptor: Request (Auth + Telemetry)
apiClient.interceptors.request.use((config) => {
    // 1. Auth Token
    const token = localStorage.getItem('token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }

    // 2. Correlation ID for Telemetry
    // WHY: Propagating a request ID allows us to trace a specific user action 
    // from the frontend button click all the way to the backend database query.
    const requestId = uuidv4();
    config.headers['X-Request-ID'] = requestId;
    config.headers['X-Correlation-ID'] = requestId;

    return config;
});

// Interceptor: Response (Error Handling)
apiClient.interceptors.response.use(
    (response) => response,
    (error: AxiosError) => {
        let message = 'An unexpected error occurred';
        let statusCode = 500;
        let details = null;

        if (error.response) {
            // Server responded with error code
            statusCode = error.response.status;
            const data = error.response.data as any;
            message = data?.message || error.message;
            details = data?.detail;

            // Log critical errors
            if (statusCode >= 500) {
                console.error(`🚨 Server Error [${statusCode}]:`, message, details);
            } else if (statusCode === 401) {
                console.warn('Authentication token expired or invalid.');
                // Optional: Trigger logout or refresh flow here
            }
        } else if (error.request) {
            // Request made but no response received (Network Error)
            message = 'Network Error: Unable to reach the server.';
            console.error('🌐 Network Error:', message);
        } else {
            // Error setting up the request
            message = error.message;
        }

        return Promise.reject(new ApiError(message, statusCode, details));
    }
);
