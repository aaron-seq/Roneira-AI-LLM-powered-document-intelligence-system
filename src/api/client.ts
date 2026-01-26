import axios from 'axios';

// Environment variable input
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Interceptor for Auth Token (if exists)
apiClient.interceptors.request.use((config) => {
    const token = localStorage.getItem('token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

// Interceptor for Error Handling (Telemetry)
apiClient.interceptors.response.use(
    (response) => response,
    (error) => {
        // Log error to monitoring service (placeholder)
        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(error);
    }
);
