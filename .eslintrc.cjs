/**
 * ESLint configuration.
 *
 * `npm run lint` was declared in package.json but no config file existed, so
 * the script failed with "ESLint couldn't find a configuration file" — the
 * frontend has never actually been linted.
 */
module.exports = {
    root: true,
    env: { browser: true, es2020: true },
    extends: [
        'eslint:recommended',
        'plugin:@typescript-eslint/recommended',
        'plugin:react-hooks/recommended',
    ],
    parser: '@typescript-eslint/parser',
    parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
        ecmaFeatures: { jsx: true },
    },
    plugins: ['@typescript-eslint', 'react-refresh'],
    settings: { react: { version: 'detect' } },
    ignorePatterns: ['dist', 'node_modules', '.eslintrc.cjs', 'coverage'],
    rules: {
        'react-refresh/only-export-components': [
            'warn',
            { allowConstantExport: true },
        ],

        // Unused variables are a real signal; an underscore prefix is the
        // documented way to say "intentionally ignored".
        '@typescript-eslint/no-unused-vars': [
            'error',
            { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
        ],

        // `any` is a warning rather than an error: several existing pages
        // still use it, and failing the build on them would block unrelated
        // work. Tighten to 'error' once those are typed.
        '@typescript-eslint/no-explicit-any': 'warn',

        // console.error/warn are legitimate in a browser client; a stray
        // console.log is usually a forgotten debug statement.
        'no-console': ['warn', { allow: ['warn', 'error', 'info'] }],
    },
};
