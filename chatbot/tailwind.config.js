/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./templates/**/*.{html,js}'],
  theme: {
    extend: {
      fontFamily: {
        'roboto': ['roboto', 'sans-serif'],
        'opensans': ['Open Sans', 'sans-serif']
      },
      fontSize: {
        xs: '0.625em',
        sm: '0.75em',
        md: '0.875em',
        base: '1em',
        'xl': '1.125em',
        '2xl': '1.25em',
        '3xl': '1.75em',
        '4xl': '2em',
        '5xl': '2.25em',
        '6xl': '2.375em',
        '7xl': '3em',
        '8xl': '3.5em',
      },
    },
  },
  plugins: [],
}

