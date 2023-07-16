import React from "react";
const styles = require('../styles.css');

export default class LabelAsPoint extends React.Component {
    onClick = () => {
        const { index, key, payload } = this.props;
        // you can do anything with the key/payload
    }
    render() {
        const { x, y } = this.props;
        return (
            <circle
                className={`${styles.dot}`}
                onClick={this.onClick}
                cx={x}
                cy={y}
                r={8}
                fill="transparent"/>
        );
    }
}