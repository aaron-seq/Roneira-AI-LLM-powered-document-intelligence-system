import { useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import {
    Box,
    Drawer,
    AppBar,
    Toolbar,
    List,
    Typography,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    CssBaseline,
    IconButton,
    useMediaQuery,
    useTheme,
    Divider,
    Tooltip,
} from '@mui/material';
import {
    Dashboard as DashboardIcon,
    UploadFile as UploadFileIcon,
    ForumOutlined as ChatIcon,
    Menu as MenuIcon,
    LogoutOutlined as LogoutIcon,
} from '@mui/icons-material';
import { microLabel } from '../../../theme';
import { useAuth } from '../../contexts/AuthContext';

const DRAWER_WIDTH = 232;

// Chat was reachable only by typing the URL: the product's central feature had
// no way in. Ordered by how the work actually goes — add documents, ask about
// them, review what came back.
const NAV = [
    { text: 'Upload', icon: <UploadFileIcon fontSize="small" />, path: '/upload' },
    { text: 'Ask', icon: <ChatIcon fontSize="small" />, path: '/chat' },
    { text: 'Documents', icon: <DashboardIcon fontSize="small" />, path: '/dashboard' },
];

const Layout = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const theme = useTheme();
    const isDesktop = useMediaQuery(theme.breakpoints.up('md'));
    const [mobileOpen, setMobileOpen] = useState(false);
    const { user, logout } = useAuth();

    const go = (path: string) => {
        navigate(path);
        setMobileOpen(false);
    };

    const nav = (
        <>
            <Toolbar />
            <Box sx={{ px: 2, pt: 2, pb: 1 }}>
                <Typography sx={microLabel}>Workspace</Typography>
            </Box>
            <List sx={{ px: 0 }}>
                {NAV.map((item) => (
                    <ListItem key={item.text} disablePadding sx={{ mb: 0.25 }}>
                        <ListItemButton
                            onClick={() => go(item.path)}
                            selected={location.pathname.startsWith(item.path)}
                        >
                            <ListItemIcon sx={{ minWidth: 34 }}>{item.icon}</ListItemIcon>
                            <ListItemText
                                primary={item.text}
                                primaryTypographyProps={{ fontSize: '0.875rem', fontWeight: 500 }}
                            />
                        </ListItemButton>
                    </ListItem>
                ))}
            </List>
            <Box sx={{ flexGrow: 1 }} />
            <Divider />
            <Box sx={{ p: 2 }}>
                <Typography sx={microLabel}>Signed in</Typography>
                <Typography variant="body2" sx={{ mt: 0.5 }}>
                    {user?.username ?? 'demo'}
                </Typography>
            </Box>
        </>
    );

    return (
        <Box sx={{ display: 'flex', minHeight: '100vh' }}>
            <CssBaseline />

            {/* Keyboard users should not have to tab through the whole sidebar
                on every page to reach the content. */}
            <Box
                component="a"
                href="#main"
                sx={{
                    position: 'absolute',
                    left: -9999,
                    zIndex: 2000,
                    '&:focus': {
                        left: 8,
                        top: 8,
                        px: 2,
                        py: 1,
                        bgcolor: 'background.paper',
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: 1,
                        color: 'text.primary',
                    },
                }}
            >
                Skip to content
            </Box>

            <AppBar position="fixed" sx={{ zIndex: theme.zIndex.drawer + 1 }}>
                <Toolbar sx={{ gap: 1, minHeight: { xs: 56, md: 60 } }}>
                    {!isDesktop && (
                        <IconButton
                            edge="start"
                            onClick={() => setMobileOpen((open) => !open)}
                            aria-label="Open navigation"
                        >
                            <MenuIcon />
                        </IconButton>
                    )}
                    <Typography
                        component="div"
                        sx={{ fontWeight: 600, letterSpacing: '-0.01em', fontSize: '0.9375rem' }}
                    >
                        Roneira
                    </Typography>
                    <Typography
                        sx={{ ...microLabel, display: { xs: 'none', sm: 'block' }, mt: '2px' }}
                    >
                        Document Intelligence
                    </Typography>
                    <Box sx={{ flexGrow: 1 }} />
                    <Tooltip title="Sign out">
                        <IconButton onClick={logout} aria-label="Sign out" size="small">
                            <LogoutIcon fontSize="small" />
                        </IconButton>
                    </Tooltip>
                </Toolbar>
            </AppBar>

            {/* Permanent on desktop, temporary on mobile. The previous drawer
                was permanent at every width, so on a phone it took a third of
                the screen and could not be dismissed. */}
            <Drawer
                variant={isDesktop ? 'permanent' : 'temporary'}
                open={isDesktop || mobileOpen}
                onClose={() => setMobileOpen(false)}
                ModalProps={{ keepMounted: true }}
                sx={{
                    width: { md: DRAWER_WIDTH },
                    flexShrink: 0,
                    '& .MuiDrawer-paper': {
                        width: DRAWER_WIDTH,
                        boxSizing: 'border-box',
                        display: 'flex',
                        flexDirection: 'column',
                    },
                }}
            >
                {nav}
            </Drawer>

            <Box
                component="main"
                id="main"
                sx={{
                    flexGrow: 1,
                    minWidth: 0, // lets wide tables and code blocks scroll rather than push the page
                    p: { xs: 2, md: 3 },
                }}
            >
                <Toolbar />
                <Outlet />
            </Box>
        </Box>
    );
};

export default Layout;
