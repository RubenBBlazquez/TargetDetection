import React, { Component } from "react";
import { render } from "react-dom";
import TrainModel from "./TrainModel";

export default class App extends Component {
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <div>
        <TrainModel />
      </div>
    );
  }
}

const appDiv = document.getElementById("app");
render(<App />, appDiv);