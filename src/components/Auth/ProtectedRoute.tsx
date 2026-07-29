import React, { ReactNode } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { Box, CircularProgress } from '@mui/material';

import { useAuth } from '@/contexts/AuthContext';

/**
 * Gate a route behind a valid session.
 *
 * Previously this hard-coded `isAuthenticated = true`, so protected pages
 * rendered for signed-out visitors and then filled with failed requests once
 * the API began enforcing authentication.
 */
const ProtectedRoute = ({ children }: { children: ReactNode }) => {
    const { isAuthenticated, isLoading } = useAuth();
    const location = useLocation();

    // Redirecting before the stored token has been validated would bounce a
    // signed-in user to the login page on every refresh.
    if (isLoading) {
        return (
            <Box
                sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minHeight: '60vh',
                }}
                role="status"
                aria-live="polite"
                aria-label="Checking your session"
            >
                <CircularProgress />
            </Box>
        );
    }

    if (!isAuthenticated) {
        // `state.from` lets the login page send the user back where they were
        // headed instead of dumping them on the dashboard.
        return <Navigate to="/login" replace state={{ from: location }} />;
    }

    return <>{children}</>;
};

export default ProtectedRoute;
