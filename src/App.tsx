import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, CssBaseline, GlobalStyles } from '@mui/material';
import { ErrorBoundary } from 'react-error-boundary';
import { Toaster } from 'react-hot-toast';

import { theme } from '../theme';
import { globalStyles } from '@/styles/globalStyles';

import { AuthProvider } from '@/contexts/AuthContext';

import Layout from '@/components/Layout/Layout';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import ErrorFallback from '@/components/Error/ErrorFallback';
import ProtectedRoute from '@/components/Auth/ProtectedRoute';

const Dashboard = lazy(() => import('@/pages/Dashboard/Dashboard'));
const DocumentUpload = lazy(() => import('@/pages/DocumentUpload/DocumentUpload'));
const DocumentViewer = lazy(() => import('@/pages/DocumentViewer/DocumentViewer'));
const AIChat = lazy(() => import('@/pages/AIChat/AIChat'));
const Login = lazy(() => import('@/pages/Auth/Login'));
const NotFound = lazy(() => import('@/pages/NotFound/NotFound'));

function App() {
  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <GlobalStyles styles={globalStyles} />
        <AuthProvider>
          <Router>
            <div className="app">
              <Routes>
                <Route path="/login" element={<Suspense fallback={<LoadingSpinner />}><Login /></Suspense>} />
                {/* All document and chat routes require authentication. */}
                <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
                  <Route index element={<Navigate to="/dashboard" replace />} />
                  <Route path="dashboard" element={<Suspense fallback={<LoadingSpinner />}><Dashboard /></Suspense>} />
                  <Route path="upload" element={<Suspense fallback={<LoadingSpinner />}><DocumentUpload /></Suspense>} />
                  <Route path="chat" element={<Suspense fallback={<LoadingSpinner />}><AIChat /></Suspense>} />
                  <Route path="documents/:documentId" element={<Suspense fallback={<LoadingSpinner />}><DocumentViewer /></Suspense>} />
                </Route>
                <Route path="*" element={<Suspense fallback={<LoadingSpinner />}><NotFound /></Suspense>} />
              </Routes>
            </div>
          </Router>
          <Toaster position="top-right" />
        </AuthProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
