import { createTheme, alpha } from '@mui/material/styles';

// ==============================================================================
// Roneira — "archive"
//
// This is an instrument for checking claims against documents, used where
// being wrong is expensive. So it borrows from archives and annotated legal
// paper rather than from dashboards: warm ink and vellum instead of slate and
// neon, hairline rules instead of glass, and one reserved accent.
//
// The previous theme was indigo-and-purple glassmorphism on Tailwind slate —
// competent, and indistinguishable from every other AI product. Worse, it
// spent colour on decoration, which left nothing to say "this passage is
// verified" with.
//
// Two rules hold the whole system together:
//
//   1. OCHRE MEANS PROVENANCE. The accent is reserved for grounding and
//      citations. If something is ochre, it is a claim you can check.
//      Nothing decorative may use it.
//   2. EVIDENCE IS MONOSPACED. Verbatim material from a document — quoted
//      passages, page numbers, checksums, IDs — is always mono. The app's own
//      words are always sans. You can tell them apart without reading.
//
// No webfonts. The product runs offline by design, and a font CDN would be a
// network dependency that fails exactly when someone is working on a plane.
// Character comes from scale, weight and tracking instead.
// ==============================================================================

const palette = {
  // Printer's ink: near-black with a warm cast, not blue-grey.
  ink: '#0F100E',
  inkRaised: '#171815',
  inkSunken: '#0A0B09',
  // Warm off-white. Long reading sessions; paper, not screen-blue.
  vellum: '#E9E5DA',
  graphite: '#95907F',
  rule: '#2A2B26',
  // Reserved: provenance, citations, verified grounding.
  ochre: '#D0A215',
  ochreDim: '#8A6D0E',
  // Annotation red — degraded modes, destructive actions.
  oxide: '#C25E3F',
  moss: '#7A9A5B',
  slate: '#5E7A8A',
};

const SANS = [
  'system-ui',
  '-apple-system',
  'Segoe UI',
  'Roboto',
  'Helvetica Neue',
  'Arial',
  'sans-serif',
].join(',');

// Evidence face. ui-monospace resolves to SF Mono / Cascadia / Consolas, all
// of which are already on the machine.
const MONO = [
  'ui-monospace',
  'SFMono-Regular',
  'SF Mono',
  'Cascadia Mono',
  'Consolas',
  'Liberation Mono',
  'monospace',
].join(',');

export const evidenceFont = MONO;

/** Verbatim document text. The signature treatment — use it for anything the
 *  document said, never for anything the interface says. */
export const evidence = {
  fontFamily: MONO,
  fontSize: '0.8125rem',
  lineHeight: 1.65,
  color: palette.vellum,
  borderLeft: `2px solid ${palette.ochreDim}`,
  paddingLeft: 12,
  whiteSpace: 'pre-wrap' as const,
};

/** Small uppercase label. Archival vernacular: wide tracking, low contrast. */
export const microLabel = {
  fontFamily: SANS,
  fontSize: '0.6875rem',
  fontWeight: 600,
  letterSpacing: '0.09em',
  textTransform: 'uppercase' as const,
  color: palette.graphite,
};

/** Panels are defined by a hairline rule, not by a shadow or a blur. */
export const panel = {
  background: palette.inkRaised,
  border: `1px solid ${palette.rule}`,
  borderRadius: 6,
};

export const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: palette.ochre, dark: palette.ochreDim, contrastText: palette.ink },
    secondary: { main: palette.slate, contrastText: palette.vellum },
    success: { main: palette.moss },
    warning: { main: palette.oxide },
    error: { main: palette.oxide },
    info: { main: palette.slate },
    background: { default: palette.ink, paper: palette.inkRaised },
    text: { primary: palette.vellum, secondary: palette.graphite },
    divider: palette.rule,
  },

  shape: { borderRadius: 6 },

  typography: {
    fontFamily: SANS,
    // Display sizes are tightened; body stays comfortable. The contrast
    // between the two is where the personality lives without a webfont.
    h1: { fontSize: '2.25rem', fontWeight: 600, letterSpacing: '-0.025em', lineHeight: 1.15 },
    h2: { fontSize: '1.75rem', fontWeight: 600, letterSpacing: '-0.02em', lineHeight: 1.2 },
    h3: { fontSize: '1.375rem', fontWeight: 600, letterSpacing: '-0.015em' },
    h4: { fontSize: '1.175rem', fontWeight: 600, letterSpacing: '-0.01em' },
    h5: { fontSize: '1.0625rem', fontWeight: 600 },
    h6: { fontSize: '0.9375rem', fontWeight: 600, letterSpacing: '0.005em' },
    body1: { fontSize: '0.9375rem', lineHeight: 1.65 },
    body2: { fontSize: '0.875rem', lineHeight: 1.6 },
    caption: { fontSize: '0.75rem', lineHeight: 1.5, color: palette.graphite },
    button: { textTransform: 'none', fontWeight: 600, letterSpacing: '0.01em' },
  },

  components: {
    MuiCssBaseline: {
      styleOverrides: {
        // Keyboard focus must always be visible, and it uses the accent
        // because "where am I" is a question about provenance too.
        '*:focus-visible': {
          outline: `2px solid ${palette.ochre}`,
          outlineOffset: 2,
          borderRadius: 3,
        },
        '::selection': {
          background: alpha(palette.ochre, 0.3),
          color: palette.vellum,
        },
        // Respect the OS setting rather than animating regardless.
        '@media (prefers-reduced-motion: reduce)': {
          '*': {
            animationDuration: '0.01ms !important',
            transitionDuration: '0.01ms !important',
            scrollBehavior: 'auto !important',
          },
        },
        body: { backgroundColor: palette.ink },
        // Scrollbars that belong to the palette instead of the OS default.
        '*::-webkit-scrollbar': { width: 10, height: 10 },
        '*::-webkit-scrollbar-track': { background: palette.inkSunken },
        '*::-webkit-scrollbar-thumb': {
          background: palette.rule,
          borderRadius: 5,
          border: `2px solid ${palette.inkSunken}`,
        },
        '*::-webkit-scrollbar-thumb:hover': { background: palette.graphite },
      },
    },

    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none', // MUI's default elevation tint
          border: `1px solid ${palette.rule}`,
        },
      },
    },

    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: palette.inkSunken,
          borderBottom: `1px solid ${palette.rule}`,
          boxShadow: 'none',
          backgroundImage: 'none',
        },
      },
    },

    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: palette.inkSunken,
          borderRight: `1px solid ${palette.rule}`,
          backgroundImage: 'none',
        },
      },
    },

    MuiButton: {
      defaultProps: { disableElevation: true },
      styleOverrides: {
        root: { borderRadius: 5, paddingInline: 14 },
        containedPrimary: {
          color: palette.ink,
          '&:hover': { backgroundColor: palette.ochre, filter: 'brightness(1.08)' },
        },
        outlined: { borderColor: palette.rule },
      },
    },

    MuiChip: {
      styleOverrides: {
        root: { borderRadius: 4, fontWeight: 500 },
        // Page numbers, scores and IDs all ride in chips, and all of them are
        // things the document said.
        labelSmall: { fontFamily: MONO, fontSize: '0.6875rem' },
        outlined: { borderColor: palette.rule },
      },
    },

    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: 5,
          marginInline: 8,
          '&.Mui-selected': {
            backgroundColor: alpha(palette.ochre, 0.14),
            '&:hover': { backgroundColor: alpha(palette.ochre, 0.2) },
          },
        },
      },
    },

    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: palette.inkSunken,
          border: `1px solid ${palette.rule}`,
          fontSize: '0.75rem',
          lineHeight: 1.5,
          maxWidth: 320,
          padding: '8px 10px',
        },
      },
    },

    MuiAlert: {
      styleOverrides: {
        root: { border: `1px solid ${palette.rule}`, borderRadius: 5 },
        standardWarning: { backgroundColor: alpha(palette.oxide, 0.12) },
        standardSuccess: { backgroundColor: alpha(palette.moss, 0.12) },
        standardInfo: { backgroundColor: alpha(palette.slate, 0.12) },
      },
    },

    MuiOutlinedInput: {
      styleOverrides: {
        notchedOutline: { borderColor: palette.rule },
      },
    },

    MuiLinearProgress: {
      styleOverrides: {
        root: { height: 3, borderRadius: 2, backgroundColor: palette.rule },
      },
    },

    MuiDivider: { styleOverrides: { root: { borderColor: palette.rule } } },
  },
});

export { palette };
export default theme;
