# Design Tokens Documentation

This document describes the design token system implemented in `styles.css` to homogenize the FastAPI frontend.

## What are Design Tokens?

Design tokens are centralized design decisions stored as CSS custom properties (variables). They provide:
- **Consistency**: Unified design language across the entire application
- **Maintainability**: Update values in one place, reflected everywhere
- **Scalability**: Easy to extend and modify the design system
- **Theme Support**: Foundation for implementing dark mode or custom themes

## Token Categories

### Colors

#### Primary
- `--color-primary: #4a90e2` - Main brand color (buttons, links, active states)
- `--color-primary-hover: #357abd` - Darker shade for hover states
- `--color-primary-light: #e3f2fd` - Light tint for backgrounds

#### Secondary
- `--color-secondary: #6c757d` - Secondary actions
- `--color-secondary-hover: #5a6268` - Secondary hover state

#### Success
- `--color-success: #5cb85c` - Success messages, assistant responses

#### Danger
- `--color-danger: #dc3545` - Error states, destructive actions
- `--color-danger-hover: #c82333` - Danger hover state
- `--color-danger-light: #ff5252` - Light danger (delete buttons)
- `--color-danger-light-hover: #ff1744` - Light danger hover

#### Text Colors
- `--color-text-primary: #333` - Main text
- `--color-text-secondary: #666` - Secondary text
- `--color-text-muted: #888` - Muted/disabled text
- `--color-text-white: white` - White text

#### Background Colors
- `--color-bg-body: #f5f5f5` - Body background
- `--color-bg-white: white` - White backgrounds
- `--color-bg-light: #f8f9fa` - Light gray backgrounds
- `--color-bg-hover: #e0e0e0` - Hover state backgrounds
- `--color-bg-disabled: #ccc` - Disabled state backgrounds

#### Border Colors
- `--color-border: #e0e0e0` - Default border color
- `--color-border-light: #f1f1f1` - Light borders

#### Semantic Colors
- `--color-info-bg: #e3f2fd` - Info message background
- `--color-info-text: #1976d2` - Info message text

#### Entity Colors (NER)
- Person: `--color-entity-per-bg`, `--color-entity-per-text`
- Organization: `--color-entity-org-bg`, `--color-entity-org-text`
- Location: `--color-entity-loc-bg`, `--color-entity-loc-text`
- Miscellaneous: `--color-entity-misc-bg`, `--color-entity-misc-text`

### Spacing

Based on an 4px base unit:
- `--space-xs: 4px` - Extra small spacing
- `--space-sm: 8px` - Small spacing
- `--space-md: 10px` - Medium spacing
- `--space-base: 12px` - Base spacing
- `--space-lg: 15px` - Large spacing
- `--space-xl: 20px` - Extra large spacing
- `--space-2xl: 25px` - 2x extra large
- `--space-3xl: 30px` - 3x extra large
- `--space-4xl: 60px` - 4x extra large

### Typography

#### Font Sizes
- `--font-size-xs: 10px` - Extra small
- `--font-size-sm: 11px` - Small
- `--font-size-base: 12px` - Base
- `--font-size-md: 13px` - Medium
- `--font-size-regular: 14px` - Regular (body text)
- `--font-size-lg: 16px` - Large
- `--font-size-xl: 18px` - Extra large
- `--font-size-2xl: 20px` - 2x large
- `--font-size-3xl: 24px` - 3x large (headings)
- `--font-size-4xl: 30px` - 4x large

#### Font Weights
- `--font-weight-normal: 400` - Normal text
- `--font-weight-medium: 500` - Medium emphasis
- `--font-weight-semibold: 600` - Headings, emphasis
- `--font-weight-bold: bold` - Strong emphasis

#### Line Heights
- `--line-height-tight: 1` - Tight spacing
- `--line-height-normal: 1.6` - Normal readable spacing

#### Font Families
- `--font-family-base` - System font stack
- `--font-family-mono: monospace` - Monospace for code

### Borders

#### Border Widths
- `--border-width-thin: 1px` - Thin borders
- `--border-width-base: 2px` - Standard borders
- `--border-width-thick: 3px` - Thick borders

#### Border Radius
- `--radius-xs: 2px` - Extra small
- `--radius-sm: 4px` - Small
- `--radius-base: 6px` - Base
- `--radius-md: 8px` - Medium (most common)
- `--radius-lg: 12px` - Large
- `--radius-full: 50%` - Full circle

### Effects

#### Shadows
- `--shadow-sm: 0 0 20px rgba(0, 0, 0, 0.25)` - Small shadow

#### Transitions
- `--transition-fast: 0.2s` - Fast animations
- `--transition-base: 0.3s` - Standard animations

### Layout

- `--sidebar-width: 300px` - Desktop sidebar width
- `--sidebar-mobile-width: 80vw` - Mobile sidebar width
- `--sidebar-mobile-max-width: 320px` - Mobile sidebar max
- `--settings-panel-width: 320px` - Settings panel width
- `--container-max-width: 900px` - Content max width
- `--ocr-preview-max-height: 400px` - OCR preview height

### Z-Index

- `--z-header: 10` - Header elements
- `--z-backdrop: 999` - Modal backdrop
- `--z-sidebar: 1000` - Sidebar (highest)

### Breakpoints

- `--breakpoint-mobile: 900px` - Mobile breakpoint (reference only)

## Usage Examples

### Using Tokens in CSS

```css
/* Use tokens instead of hardcoded values */
.my-button {
    padding: var(--space-md) var(--space-xl);
    background: var(--color-primary);
    color: var(--color-text-white);
    border-radius: var(--radius-base);
    font-size: var(--font-size-regular);
    transition: all var(--transition-fast);
}

.my-button:hover {
    background: var(--color-primary-hover);
}
```

### Creating a New Component

```css
.new-card {
    /* Spacing */
    padding: var(--space-xl);
    margin-bottom: var(--space-lg);
    
    /* Colors */
    background: var(--color-bg-white);
    border: var(--border-width-thin) solid var(--color-border);
    
    /* Effects */
    border-radius: var(--radius-md);
    transition: all var(--transition-fast);
}

.new-card:hover {
    border-color: var(--color-primary);
    background: var(--color-bg-light);
}
```

## Modifying the Design System

### Changing Colors

To change the primary color throughout the app:

```css
:root {
    --color-primary: #your-new-color;
    --color-primary-hover: #darker-shade;
    --color-primary-light: #lighter-tint;
}
```

### Adjusting Spacing

To increase overall spacing:

```css
:root {
    --space-base: 16px; /* was 12px */
    --space-lg: 20px;   /* was 15px */
    /* etc. */
}
```

### Adding New Tokens

When adding new tokens, follow the naming convention:

```css
:root {
    /* Category + Specificity + Property */
    --color-warning-bg: #fff3cd;
    --color-warning-text: #856404;
    
    --space-5xl: 80px;
    
    --font-size-5xl: 36px;
}
```

## Implementing Dark Mode

The token system makes dark mode implementation straightforward:

```css
@media (prefers-color-scheme: dark) {
    :root {
        --color-bg-body: #1a1a1a;
        --color-bg-white: #2d2d2d;
        --color-bg-light: #363636;
        --color-text-primary: #e0e0e0;
        --color-text-secondary: #b0b0b0;
        --color-border: #404040;
        /* etc. */
    }
}
```

## Benefits of This System

1. **Single Source of Truth**: All design decisions in one place
2. **Easy Global Updates**: Change primary color once, updates everywhere
3. **Consistency**: Prevents random one-off values
4. **Developer Experience**: Autocomplete in modern IDEs
5. **Performance**: CSS variables are performant
6. **Maintainability**: Clear naming makes code self-documenting
7. **Future-Proof**: Easy to add themes, dark mode, accessibility features

## Best Practices

1. **Always use tokens** for colors, spacing, typography
2. **Don't hardcode values** unless absolutely necessary
3. **Follow naming conventions** when adding new tokens
4. **Group related tokens** in the :root declaration
5. **Document new tokens** in this file
6. **Test changes** across all components
7. **Consider accessibility** when modifying color tokens

## Migration Guide

When adding new styles to the application:

**Before:**
```css
.new-element {
    padding: 15px;
    color: #666;
    font-size: 14px;
    border-radius: 8px;
}
```

**After:**
```css
.new-element {
    padding: var(--space-lg);
    color: var(--color-text-secondary);
    font-size: var(--font-size-regular);
    border-radius: var(--radius-md);
}
```

## Questions?

For questions or suggestions about the design token system, please refer to the main `styles.css` file or update this documentation accordingly.
