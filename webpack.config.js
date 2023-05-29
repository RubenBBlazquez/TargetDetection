const path = require("path");
const webpack = require("webpack");

module.exports = {
    entry:  [__dirname + '/DataTraining/react_components'],
    output: {
        path: path.resolve(__dirname, "static/react_build"),
    },
    module: {
        rules: [
            {
                test: /\.js$/,
                exclude: /node_modules/,
                use: {
                    loader: "babel-loader",
                },
            },
        ],
    },
    optimization: {
        minimize: true,
    },
};