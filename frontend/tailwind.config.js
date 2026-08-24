/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        charcoal: {
          950: '#070A0E',
          900: '#0B0F14',
          850: '#10161E',
          800: '#151D28',
          750: '#1C2634',
          700: '#233041',
          600: '#33445C',
          500: '#4D6282',
        },
        rf: {
          green: {
            DEFAULT: '#10B981',
            glow: '#059669',
            light: '#34D399',
            dark: '#064E3B',
            bg: 'rgba(16, 185, 129, 0.12)',
            border: 'rgba(16, 185, 129, 0.35)',
          },
          cyan: {
            DEFAULT: '#06B6D4',
            glow: '#0891B2',
            light: '#38BDF8',
            dark: '#164E63',
            bg: 'rgba(6, 182, 212, 0.12)',
            border: 'rgba(6, 182, 212, 0.35)',
          },
          amber: {
            DEFAULT: '#F59E0B',
            glow: '#D97706',
            light: '#FBBF24',
            dark: '#78350F',
            bg: 'rgba(245, 158, 11, 0.12)',
            border: 'rgba(245, 158, 11, 0.35)',
          },
          red: {
            DEFAULT: '#EF4444',
            glow: '#DC2626',
            light: '#F87171',
            dark: '#7F1D1D',
            bg: 'rgba(239, 68, 68, 0.12)',
            border: 'rgba(239, 68, 68, 0.35)',
          },
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
        display: ['Outfit', 'sans-serif'],
      },
      boxShadow: {
        'glow-green': '0 0 20px -3px rgba(16, 185, 129, 0.35)',
        'glow-cyan': '0 0 20px -3px rgba(6, 182, 212, 0.35)',
        'glow-amber': '0 0 20px -3px rgba(245, 158, 11, 0.35)',
        'panel': '0 4px 20px -2px rgba(0, 0, 0, 0.5), inset 0 1px 0 0 rgba(255, 255, 255, 0.05)',
      },
      animation: {
        'pulse-subtle': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'flow-right': 'flowRight 2s linear infinite',
      },
      keyframes: {
        flowRight: {
          '0%': { transform: 'translateX(-100%)', opacity: '0' },
          '50%': { opacity: '1' },
          '100%': { transform: 'translateX(100%)', opacity: '0' },
        },
      },
    },
  },
  plugins: [],
}
