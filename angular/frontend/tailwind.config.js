/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Fraunces', 'Georgia', 'serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      colors: {
        // Brand: iris / indigo-violet — distinct from default UI blue, and
        // doesn't collide with the red/green/amber clinical semantics.
        brand: {
          50: '#f2f1fe',
          100: '#e6e4fd',
          200: '#cfcbfb',
          300: '#b0a9f7',
          400: '#9184f0',
          500: '#7363e8',
          600: '#5f49d9',
          700: '#4f3bbe',
          800: '#42349a',
          900: '#382f7b',
        },
        ink: {
          DEFAULT: '#0d1017',
          900: '#0d1017',
          800: '#151a23',
          700: '#1f2630',
          600: '#37404d',
          500: '#5a6472',
          400: '#828c9b',
          300: '#aab2bd',
        },
        paper: '#f3f3f7',
        // keep the legacy alias so existing markup doesn't break
        clinical: {
          blue: '#5f49d9',
          alert: '#ef4444',
          safe: '#22c55e',
          bg: '#f3f3f7',
        },
      },
      boxShadow: {
        soft: '0 1px 2px rgba(13,16,23,0.04), 0 4px 16px -6px rgba(13,16,23,0.10)',
        lift: '0 2px 4px rgba(13,16,23,0.04), 0 18px 40px -12px rgba(13,16,23,0.18)',
        glow: '0 0 0 1px rgba(95,73,217,0.18), 0 12px 36px -10px rgba(95,73,217,0.40)',
        inset: 'inset 0 1px 0 0 rgba(255,255,255,0.6)',
      },
      backgroundImage: {
        'grain': "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.4'/%3E%3C/svg%3E\")",
      },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'scale-in': {
          '0%': { opacity: '0', transform: 'scale(0.96)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        shimmer: {
          '100%': { transform: 'translateX(100%)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-8px)' },
        },
        'pulse-ring': {
          '0%': { transform: 'scale(0.8)', opacity: '0.6' },
          '100%': { transform: 'scale(2.2)', opacity: '0' },
        },
      },
      animation: {
        'fade-up': 'fade-up 0.6s cubic-bezier(0.22, 1, 0.36, 1) both',
        'fade-in': 'fade-in 0.6s ease both',
        'scale-in': 'scale-in 0.5s cubic-bezier(0.22, 1, 0.36, 1) both',
        shimmer: 'shimmer 2s infinite',
        float: 'float 6s ease-in-out infinite',
        'pulse-ring': 'pulse-ring 1.8s cubic-bezier(0.22, 1, 0.36, 1) infinite',
      },
    },
  },
  plugins: [],
}
