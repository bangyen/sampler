export default [
  {
    files: ['static/**/*.js'],
    languageOptions: {
      ecmaVersion: 2021,
      sourceType: 'script',
      globals: {
        console: 'readonly',
        document: 'readonly',
        window: 'readonly',
        EventSource: 'readonly',
        AbortController: 'readonly',
        fetch: 'readonly',
        FormData: 'readonly',
        Blob: 'readonly',
        URL: 'readonly',
        setTimeout: 'readonly',
        navigator: 'readonly',
        TextDecoder: 'readonly',
        performance: 'readonly',
        alert: 'readonly',
        FileReader: 'readonly',
        File: 'readonly'
      }
    },
    rules: {
      'no-unused-vars': 'warn',
      'no-undef': 'error',
      'semi': ['error', 'always'],
      'quotes': ['error', 'single', { avoidEscape: true }]
    }
  }
];
