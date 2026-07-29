import React from 'react';
import { Alert, AlertTitle, Box, Button, Container, Typography } from '@mui/material';
import type { FallbackProps } from 'react-error-boundary';

/**
 * Top-level error boundary UI.
 *
 * Replaces a bare "Something went wrong." div. A user who hits a crash needs
 * two things the old version did not give them: a way to recover without
 * losing the tab, and something concrete to quote in a bug report.
 */
const ErrorFallback = ({ error, resetErrorBoundary }: Partial<FallbackProps>) => (
    <Container maxWidth="sm" sx={{ mt: 8 }}>
        <Alert severity="error" role="alert">
            <AlertTitle>Something went wrong</AlertTitle>
            <Typography variant="body2" sx={{ mb: 2 }}>
                The page failed to render. Retrying often clears it; if it keeps
                happening, please open an issue with the details below.
            </Typography>

            {error?.message && (
                <Box
                    component="pre"
                    sx={{
                        p: 1.5,
                        mb: 2,
                        borderRadius: 1,
                        bgcolor: 'action.hover',
                        fontSize: '0.75rem',
                        // Long stack frames must not force the page to scroll
                        // sideways on a phone.
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                        maxHeight: 200,
                        overflow: 'auto',
                    }}
                >
                    {error.message}
                </Box>
            )}

            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {resetErrorBoundary && (
                    <Button variant="contained" onClick={resetErrorBoundary}>
                        Try again
                    </Button>
                )}
                <Button variant="outlined" onClick={() => window.location.reload()}>
                    Reload the page
                </Button>
            </Box>
        </Alert>
    </Container>
);

export default ErrorFallback;
