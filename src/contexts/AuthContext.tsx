/**
 * Authentication state for the browser client.
 *
 * This replaces a stub that returned `null` and a ProtectedRoute that
 * hard-coded `isAuthenticated = true`, so the UI behaved as if everyone were
 * signed in while the API — now that it enforces auth — answers 401.
 */
import React, {
    createContext,
    ReactNode,
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useState,
} from 'react';

import { apiClient } from '../api/client';

export interface User {
    user_id: string;
    username: string;
    roles: string[];
    is_anonymous: boolean;
}

interface AuthContextValue {
    user: User | null;
    isAuthenticated: boolean;
    /** True until any stored token has been checked against the API. */
    isLoading: boolean;
    error: string | null;
    login: (username: string, password: string) => Promise<void>;
    logout: () => void;
}

const TOKEN_STORAGE_KEY = 'token';

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
    const [user, setUser] = useState<User | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    /**
     * Validate any stored token on mount.
     *
     * A token in localStorage is not proof of a session: it may have expired
     * while the tab was closed, or been signed with a key the server has
     * since rotated. Asking `/auth/me` is the only way to know.
     */
    useEffect(() => {
        const token = localStorage.getItem(TOKEN_STORAGE_KEY);
        if (!token) {
            setIsLoading(false);
            return;
        }

        let cancelled = false;
        apiClient
            .get<User>('/auth/me')
            .then((response) => {
                if (!cancelled) setUser(response.data);
            })
            .catch(() => {
                if (!cancelled) {
                    localStorage.removeItem(TOKEN_STORAGE_KEY);
                    setUser(null);
                }
            })
            .finally(() => {
                if (!cancelled) setIsLoading(false);
            });

        return () => {
            cancelled = true;
        };
    }, []);

    const login = useCallback(async (username: string, password: string) => {
        setError(null);

        // The token endpoint follows the OAuth2 password flow, so it expects
        // form encoding rather than JSON.
        const form = new URLSearchParams();
        form.append('username', username);
        form.append('password', password);

        try {
            const response = await apiClient.post('/auth/token', form, {
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            });

            localStorage.setItem(TOKEN_STORAGE_KEY, response.data.access_token);
            setUser({
                user_id: response.data.user_id,
                username: response.data.username ?? username,
                roles: response.data.roles ?? [],
                is_anonymous: false,
            });
        } catch (err) {
            const message =
                err instanceof Error ? err.message : 'Sign in failed. Please try again.';
            setError(message);
            throw err;
        }
    }, []);

    const logout = useCallback(() => {
        localStorage.removeItem(TOKEN_STORAGE_KEY);
        setUser(null);
        setError(null);
    }, []);

    const value = useMemo<AuthContextValue>(
        () => ({
            user,
            isAuthenticated: user !== null,
            isLoading,
            error,
            login,
            logout,
        }),
        [user, isLoading, error, login, logout],
    );

    return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthContextValue => {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error('useAuth must be used within an <AuthProvider>');
    }
    return context;
};
