import React from "react";
import {Card, Container} from "react-bootstrap";
import PredictionsByDateChart from "../Components/PredictionsByDateChart";
import MisfireOfLastPredictionsChart from "../Components/MisfireOfLastPredictionsChart";
import ShootsErrorOfLastPredictionsChart from "../Components/ShootsErrorOfLastPredictionsChart";
import TotalPredictionsPercentages from "../Components/TotalPredictionsPercentages";
import PredictionImagesList from "../Components/PredictionImagesList";
import Filters from "../Components/Filters";

function Predictions() {
    return (
        <div className={'d-flex justify-content-center flex-column'}>
            <Container className={'mt-3 ms-1 row d-flex align-items-center justify-content-center'} fluid>
                <Container className={'col-lg-8 col-sm-12 rounded'}>
                    <Card className={'border'} style={{height: "60px"}}>
                        <h1 className={'text-center'}>Predictions</h1>
                    </Card>
                    <Filters/>
                </Container>
                <Container
                    className={'col-lg-4 col-sm-12 mt-lg-0 mt-sm-3 border rounded d-flex justify-content-center align-items-center flex-column'}>
                    <h4 className={'text-center'}>Predicted Targets</h4>
                    <TotalPredictionsPercentages/>
                </Container>
            </Container>
            <PredictionImagesList/>
            <Container className={'mt-3 d-flex row ms-1'} fluid>
                <PredictionsByDateChart/>
                <ShootsErrorOfLastPredictionsChart/>
                <MisfireOfLastPredictionsChart/>
            </Container>
        </div>
    );
}

export default Predictions;