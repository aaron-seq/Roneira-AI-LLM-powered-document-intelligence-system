import React, { FormEvent, useState } from 'react';
import { Navigate, useLocation, useNavigate } from 'react-router-dom';
import {
    Alert,
    Box,
    Button,
    CircularProgress,
    Container,
    Paper,
    TextField,
    Typography,
} from '@mui/material';

import { useAuth } from '@/contexts/AuthContext';

interface LocationState {
    from?: { pathname: string };
}

/**
 * Sign-in page.
 *
 * The previous version rendered a form that submitted nowhere and asked for
 * an email address the API has never accepted — the credential is a username.
 */
const Login = () => {
    const { login, isAuthenticated, isLoading: isRestoringSession } = useAuth();
    const navigate = useNavigate();
    const location = useLocation();

    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    const destination = (location.state as LocationState)?.from?.pathname ?? '/dashboard';

    if (isAuthenticated) {
        return <Navigate to={destination} replace />;
    }

    const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setError(null);
        setIsSubmitting(true);

        try {
            await login(username.trim(), password);
            navigate(destination, { replace: true });
        } catch {
            // Deliberately generic: the API does not reveal whether it was the
            // username or the password that was wrong, and neither should we.
            setError('Sign in failed. Check your username and password.');
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <Container component="main" maxWidth="xs" sx={{ mt: 8 }}>
            <Paper elevation={2} sx={{ p: 4 }}>
                <Typography component="h1" variant="h5" gutterBottom>
                    Sign in
                </Typography>

                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    A local install ships with the <code>demo</code> account
                    (password <code>demo</code>). Replace it before serving real
                    documents — see docs/SECURITY.md.
                </Typography>

                {error && (
                    <Alert severity="error" sx={{ mb: 2 }} role="alert">
                        {error}
                    </Alert>
                )}

                <Box component="form" onSubmit={handleSubmit} noValidate>
                    <TextField
                        variant="outlined"
                        margin="normal"
                        required
                        fullWidth
                        id="username"
                        label="Username"
                        name="username"
                        autoComplete="username"
                        autoFocus
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        disabled={isSubmitting}
                    />
                    <TextField
                        variant="outlined"
                        margin="normal"
                        required
                        fullWidth
                        name="password"
                        label="Password"
                        type="password"
                        id="password"
                        autoComplete="current-password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        disabled={isSubmitting}
                    />
                    <Button
                        type="submit"
                        fullWidth
                        variant="contained"
                        color="primary"
                        sx={{ mt: 3 }}
                        disabled={
                            isSubmitting ||
                            isRestoringSession ||
                            !username.trim() ||
                            !password
                        }
                        startIcon={
                            isSubmitting ? (
                                <CircularProgress size={18} color="inherit" />
                            ) : undefined
                        }
                    >
                        {isSubmitting ? 'Signing in…' : 'Sign in'}
                    </Button>
                </Box>
            </Paper>
        </Container>
    );
};

export default Login;
