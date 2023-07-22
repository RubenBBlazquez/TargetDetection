import React from "react";
import {Card, Container} from "react-bootstrap";
import PredictionsByDateChart from "./Components/PredictionsByDateChart";
import MisfireOfLastPredictionsChart from "./Components/MisfireOfLastPredictionsChart";
import ShootsErrorOfLastPredictionsChart from "./Components/ShootsErrorOfLastPredictionsChart";
import TotalPredictionsPercentages from "./Components/TotalPredictionsPercentages";
import PredictionImagesList from "./Components/PredictionImagesList";
import Filters from "../CommonComponents/Filters";
import Alert from 'react-bootstrap/Alert';
import {getPredictions} from "./methods";

function Predictions() {
    const predictionsData = getPredictions();

    return (
        <div className={'d-flex justify-content-center flex-column border'}>
            <Container className={'mt-3 ms-1 row d-flex align-items-center justify-content-center'} fluid>
                <Container className={'d-flex justify-content-end'}>
                    <Alert key={'info'} variant={'warning'} className={'w-50'}>
                        All charts are interactive. You can click on the legend to hide/show the data and interact with
                        all page.
                    </Alert>
                </Container>
                <Container className={'col-lg-8 col-sm-12 rounded'}>
                    <Card className={'border'} style={{height: "60px"}}>
                        <h1 className={'text-center'}>Predictions</h1>
                    </Card>
                    <Filters/>
                    <PredictionImagesList/>
                </Container>
                <Container
                    className={'col-lg-4 col-sm-12 mt-lg-0 mt-sm-3 border rounded d-flex justify-content-center align-items-center flex-column'}>
                    <h4 className={'text-center'}>Predicted Targets</h4>
                    <TotalPredictionsPercentages/>
                </Container>
            </Container>
            <Container className={'mt-3 d-flex row ms-1'} fluid>
                <PredictionsByDateChart/>
                <ShootsErrorOfLastPredictionsChart/>
                <MisfireOfLastPredictionsChart/>
            </Container>
        </div>
    );
}

export default Predictions;