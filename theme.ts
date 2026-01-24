import { createTheme, alpha } from '@mui/material/styles';

// ==============================================================================
// Roneira AI - Modern 2026 Dark Theme
// Figma-Inspired Glassmorphism Design with Deep Slate Background
// ==============================================================================

// --- Premium Dark Color Palette ---
const colors = {
  primary: {
    main: '#6366f1', // Vibrant indigo
    light: '#818cf8',
    dark: '#4f46e5',
    contrastText: '#ffffff',
  },
  secondary: {
    main: '#8b5cf6', // Rich purple
    light: '#a78bfa',
    dark: '#7c3aed',
    contrastText: '#ffffff',
  },
  success: {
    main: '#10b981', // Emerald
    light: '#34d399',
    dark: '#059669',
  },
  warning: {
    main: '#f59e0b', // Amber
    light: '#fbbf24',
    dark: '#d97706',
  },
  error: {
    main: '#ef4444', // Red
    light: '#f87171',
    dark: '#dc2626',
  },
  info: {
    main: '#06b6d4', // Cyan accent
    light: '#22d3ee',
    dark: '#0891b2',
  },
  background: {
    default: '#0a0f1a', // Deep dark
    paper: '#0f172a',   // Slate 900
  },
  text: {
    primary: '#f1f5f9',   // Slate 100
    secondary: '#94a3b8', // Slate 400
  },
};

// --- Glassmorphism Card Styles ---
export const glassCard = {
  background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%)',
  backdropFilter: 'blur(20px)',
  WebkitBackdropFilter: 'blur(20px)',
  border: '1px solid rgba(99, 102, 241, 0.15)',
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05)',
};

// --- Glowing Accent Styles ---
export const glowEffect = {
  primary: '0 0 20px rgba(99, 102, 241, 0.4), 0 0 40px rgba(99, 102, 241, 0.2)',
  cyan: '0 0 20px rgba(6, 182, 212, 0.4), 0 0 40px rgba(6, 182, 212, 0.2)',
  success: '0 0 20px rgba(16, 185, 129, 0.4), 0 0 40px rgba(16, 185, 129, 0.2)',
};

// --- Dark Theme Configuration ---
export const theme = createTheme({
  palette: {
    mode: 'dark',
    ...colors,
  },
  
  typography: {
    fontFamily: '"Inter", "SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif',
    h1: { 
      fontSize: '2.5rem', 
      fontWeight: 700,
      letterSpacing: '-0.02em',
      color: colors.text.primary,
    },
    h2: { 
      fontSize: '2rem', 
      fontWeight: 700,
      letterSpacing: '-0.01em',
    },
    h3: { 
      fontSize: '1.5rem', 
      fontWeight: 600,
    },
    h4: {
      fontSize: '1.25rem',
      fontWeight: 600,
    },
    h5: {
      fontSize: '1rem',
      fontWeight: 600,
    },
    h6: {
      fontSize: '0.875rem',
      fontWeight: 600,
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
    },
    body1: {
      fontSize: '0.9375rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.8125rem',
      lineHeight: 1.5,
      color: colors.text.secondary,
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
      letterSpacing: '0.02em',
    },
  },

  shape: { borderRadius: 16 },

  shadows: [
    'none',
    '0 2px 8px rgba(0, 0, 0, 0.3)',
    '0 4px 12px rgba(0, 0, 0, 0.35)',
    '0 6px 16px rgba(0, 0, 0, 0.4)',
    '0 8px 24px rgba(0, 0, 0, 0.4)',
    '0 12px 32px rgba(0, 0, 0, 0.45)',
    '0 16px 40px rgba(0, 0, 0, 0.5)',
    ...Array(18).fill('none'),
  ] as any,

  components: {
    MuiCssBaseline: {
      styleOverrides: `
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {
          --gradient-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
          --gradient-cyan: linear-gradient(135deg, #06b6d4 0%, #10b981 100%);
          --glass-bg: rgba(15, 23, 42, 0.8);
          --glass-border: rgba(99, 102, 241, 0.15);
        }
        
        body {
          background: linear-gradient(180deg, #0a0f1a 0%, #0f172a 50%, #1e293b 100%);
          min-height: 100vh;
        }
        
        ::-webkit-scrollbar {
          width: 6px;
          height: 6px;
        }
        
        ::-webkit-scrollbar-track {
          background: rgba(15, 23, 42, 0.5);
        }
        
        ::-webkit-scrollbar-thumb {
          background: rgba(99, 102, 241, 0.4);
          border-radius: 3px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
          background: rgba(99, 102, 241, 0.6);
        }
        
        ::selection {
          background: rgba(99, 102, 241, 0.4);
        }
      `,
    },
    
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          padding: '12px 28px',
          fontSize: '0.9375rem',
          fontWeight: 600,
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        },
        contained: {
          background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
          boxShadow: '0 4px 16px rgba(99, 102, 241, 0.3)',
          '&:hover': {
            background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
            boxShadow: '0 8px 24px rgba(99, 102, 241, 0.4)',
            transform: 'translateY(-2px)',
          },
        },
        outlined: {
          borderColor: 'rgba(99, 102, 241, 0.5)',
          borderWidth: 2,
          '&:hover': {
            borderWidth: 2,
            borderColor: '#6366f1',
            background: 'rgba(99, 102, 241, 0.1)',
          },
        },
      },
    },
    
    MuiCard: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          border: '1px solid rgba(99, 102, 241, 0.1)',
          borderRadius: 20,
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
          transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            border: '1px solid rgba(99, 102, 241, 0.3)',
            boxShadow: '0 16px 48px rgba(0, 0, 0, 0.5), 0 0 40px rgba(99, 102, 241, 0.1)',
            transform: 'translateY(-4px)',
          },
        },
      },
    },
    
    MuiPaper: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.8) 100%)',
          backdropFilter: 'blur(16px)',
          borderRadius: 16,
        },
      },
    },
    
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(180deg, rgba(10, 15, 26, 0.95) 0%, rgba(15, 23, 42, 0.9) 100%)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgba(99, 102, 241, 0.1)',
          boxShadow: '0 4px 24px rgba(0, 0, 0, 0.3)',
        },
      },
    },
    
    MuiDrawer: {
      styleOverrides: {
        paper: {
          background: 'linear-gradient(180deg, rgba(10, 15, 26, 0.98) 0%, rgba(15, 23, 42, 0.95) 100%)',
          backdropFilter: 'blur(20px)',
          borderRight: '1px solid rgba(99, 102, 241, 0.1)',
        },
      },
    },
    
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 12,
            background: 'rgba(15, 23, 42, 0.6)',
            '& fieldset': {
              borderColor: 'rgba(99, 102, 241, 0.2)',
              borderWidth: 2,
            },
            '&:hover fieldset': {
              borderColor: 'rgba(99, 102, 241, 0.4)',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#6366f1',
              boxShadow: '0 0 20px rgba(99, 102, 241, 0.2)',
            },
          },
        },
      },
    },
    
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 500,
          background: 'rgba(99, 102, 241, 0.15)',
          border: '1px solid rgba(99, 102, 241, 0.3)',
        },
        filled: {
          background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(139, 92, 246, 0.3) 100%)',
        },
      },
    },
    
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          height: 8,
          background: 'rgba(99, 102, 241, 0.15)',
        },
        bar: {
          borderRadius: 8,
          background: 'linear-gradient(90deg, #6366f1 0%, #06b6d4 100%)',
        },
      },
    },
    
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          margin: '4px 8px',
          transition: 'all 0.3s ease',
          '&:hover': {
            background: 'rgba(99, 102, 241, 0.1)',
          },
          '&.Mui-selected': {
            background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.15) 100%)',
            borderLeft: '3px solid #6366f1',
            '&:hover': {
              background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(139, 92, 246, 0.2) 100%)',
            },
          },
        },
      },
    },
    
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          background: 'rgba(15, 23, 42, 0.95)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(99, 102, 241, 0.2)',
          borderRadius: 8,
          padding: '8px 14px',
          fontSize: '0.8125rem',
        },
      },
    },
    
    MuiFab: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
          boxShadow: '0 8px 24px rgba(99, 102, 241, 0.4)',
          '&:hover': {
            background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
            boxShadow: '0 12px 32px rgba(99, 102, 241, 0.5), 0 0 40px rgba(99, 102, 241, 0.3)',
          },
        },
      },
    },
  },
});

export default theme;