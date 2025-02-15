// Configure the require.js package to recognize plotly
requirejs.config({
    paths: {
        base: '/static/base',
        Plotly: 'https://cdn.plot.ly/plotly-latest.min.js',
    },
});