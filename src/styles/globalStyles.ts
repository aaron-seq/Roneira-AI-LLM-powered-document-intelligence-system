import { css } from '@emotion/react';

// Only what the MUI theme cannot reach. Everything else lives in theme.ts so
// there is one place to change a colour.
export const globalStyles = css`
  html {
    /* Anchors and in-page jumps land below the fixed app bar rather than
       underneath it. */
    scroll-padding-top: 80px;
  }

  body {
    margin: 0;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
  }

  /* Verbatim document material is monospaced everywhere it appears — that is
     how you tell what the document said from what the interface says. */
  code,
  pre,
  samp {
    font-family: ui-monospace, SFMono-Regular, 'SF Mono', 'Cascadia Mono',
      Consolas, 'Liberation Mono', monospace;
    font-size: 0.8125rem;
  }

  /* Long identifiers and URLs should wrap rather than widen the page. */
  pre {
    white-space: pre-wrap;
    word-break: break-word;
  }
`;
