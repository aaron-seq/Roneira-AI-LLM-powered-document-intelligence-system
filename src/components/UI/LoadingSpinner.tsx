import React from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';

interface LoadingSpinnerProps {
    /** Announced to assistive technology and shown beneath the spinner. */
    label?: string;
    fullHeight?: boolean;
}

/**
 * Loading indicator used as the Suspense fallback for lazy routes.
 *
 * `role="status"` with `aria-live="polite"` means a screen reader announces
 * the wait instead of landing on a silently empty page.
 */
const LoadingSpinner = ({
    label = 'Loading…',
    fullHeight = true,
}: LoadingSpinnerProps) => (
    <Box
        role="status"
        aria-live="polite"
        sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 2,
            minHeight: fullHeight ? '60vh' : 120,
            p: 3,
        }}
    >
        <CircularProgress aria-hidden="true" />
        <Typography variant="body2" color="text.secondary">
            {label}
        </Typography>
    </Box>
);

export default LoadingSpinner;
