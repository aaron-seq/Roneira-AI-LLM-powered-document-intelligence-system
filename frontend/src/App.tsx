import React, { Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { HelmetProvider } from 'react-helmet-async';
import { ErrorBoundary } from 'react-error-boundary';
import { Toaster } from 'react-hot-toast';

import { modernTheme } from '@/styles/theme';
import { AppProviders } from '@/providers/AppProviders';
import { Routes as AppRoutes } from '@/routes';
import { LoadingSpinner } from '@/components/ui';
import { ErrorFallback } from '@/components/error';

// Configure React Query with optimized settings
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 1,
    },
  },
});

const App: React.FC = () => {
  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <HelmetProvider>
        <QueryClientProvider client={queryClient}>
          <ThemeProvider theme={modernTheme}>
            <CssBaseline />
            <AppProviders>
              <BrowserRouter>
                <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
                  <Suspense fallback={<LoadingSpinner />}>
                    <AppRoutes />
                  </Suspense>
                  
                  <Toaster
                    position="top-right"
                    toastOptions={{
                      duration: 4000,
                      style: {
                        background: '#363636',
                        color: '#fff',
                      },
                    }}
                  />
                </div>
              </BrowserRouter>
            </AppProviders>
          </ThemeProvider>
          
          {import.meta.env.DEV && (
            <ReactQueryDevtools initialIsOpen={false} />
          )}
        </QueryClientProvider>
      </HelmetProvider>
    </ErrorBoundary>
  );
};

export default App;
