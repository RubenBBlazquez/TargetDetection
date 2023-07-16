import React from "react";
import {Card, Container} from "react-bootstrap";
import PredictionsByDateChart from "../Components/PredictionsByDateChart";
import MisfireOfLastPredictionsChart from "../Components/MisfireOfLastPredictionsChart";
import ShootsErrorOfLastPredictionsChart from "../Components/ShootsErrorOfLastPredictionsChart";
import {Form} from "react-bootstrap";
import TotalPredictionsPercentages from "../Components/TotalPredictionsPercentages";

function Predictions() {
    const [linearChartByPD, setLinearChartByPD] = React.useState(true);
    const [misfireBarChart, setMisfireBarChart] = React.useState(true);
    const [shootsErrorBarChart, setShootsErrorBarChart] = React.useState(true);

    return (
        <div className={'d-flex justify-content-center flex-column'}>
            <Container className={'mt-3 ms-1 row d-flex align-items-center justify-content-center'} fluid>
                <Container className={'col-lg-8 col-sm-12 rounded'}>
                    <Card className={'border'} style={{height: "60px"}}>
                        <h1 className={'text-center'}>Predictions</h1>
                    </Card>
                </Container>
                <Container className={'col-lg-4 col-sm-12 mt-lg-0 mt-sm-3 border rounded d-flex justify-content-center align-items-center flex-column'}>
                    <h3 className={'text-center'}>Predicted Targets</h3>
                    <TotalPredictionsPercentages/>
                </Container>
            </Container>

            <Container className={'mt-5 d-flex row ms-1'} fluid>
                <Container className={'col-lg-4 col-sm-12 mt-lg-0 mt-sm-3 border rounded'}>
                    <div className={'d-flex flex-column align-items-center'}>
                        <h3 className={'text-center'}>Predictions By Date</h3>
                        <div className={'w-100 p-2'}>
                            <Form.Check
                                type="switch"
                                label="Change to Linear Chart"
                                id="set-linear-chart-predictions-by-date"
                                className={`float-end ${linearChartByPD && 'd-none'}`}
                                checked={true}
                                onChange={() => {
                                    setLinearChartByPD(true);
                                }}
                            />
                            <Form.Check
                                type="switch"
                                label="Change to Bar Chart"
                                id="set-bar-chart-predictions-by-date"
                                checked={false}
                                className={`float-end ${!linearChartByPD && 'd-none'}`}
                                onChange={() => {
                                    setLinearChartByPD(false);
                                }}
                            />
                        </div>
                    </div>
                    <PredictionsByDateChart useLineChart={linearChartByPD}/>
                </Container>
                <Container className={'col-lg-4 col-xs-12 mt-lg-0 mt-sm-3 border rounded'}>
                    <div className={'d-flex flex-column align-items-center'}>
                        <h3 className={'text-center'}>Shoots Error of last predictions</h3>
                        <div className={'w-100 p-2'}>
                            <Form.Check
                                type="switch"
                                label="Change to Linear Chart"
                                id="set-linear-chart-shoots-error"
                                className={`float-end ${!shootsErrorBarChart && 'd-none'}`}
                                checked={false}
                                onChange={() => {
                                    setShootsErrorBarChart(false);
                                }}
                            />
                            <Form.Check
                                type="switch"
                                label="Change to Bar Chart"
                                id="set-bar-chart-shoots-error"
                                className={`float-end ${shootsErrorBarChart && 'd-none'}`}
                                checked={true}
                                onChange={() => {
                                    setShootsErrorBarChart(true);
                                }}
                            />
                        </div>
                    </div>
                    <ShootsErrorOfLastPredictionsChart useBarChart={shootsErrorBarChart}/>
                </Container>
                <Container className={'col-lg-4 col-xs-12 mt-lg-0 mt-sm-3 border rounded'}>
                    <div className={'d-flex flex-column align-items-center'}>
                        <h3 className={'text-center'}>Misfire of last predictions</h3>
                        <div className={'w-100 p-2'}>
                            <Form.Check
                                type="switch"
                                label="Change to Linear Chart"
                                id="set-linear-chart-misfires"
                                className={`float-end ${!misfireBarChart && 'd-none'}`}
                                checked={false}
                                onChange={() => {
                                    setMisfireBarChart(false);
                                }}
                            />
                            <Form.Check
                                type="switch"
                                label="Change to Bar Chart"
                                id="set-bar-chart-misfires"
                                className={`float-end ${misfireBarChart && 'd-none'}`}
                                checked={true}
                                onChange={() => {
                                    setMisfireBarChart(true);
                                }}
                            />
                        </div>
                    </div>
                    <MisfireOfLastPredictionsChart useBarChart={misfireBarChart}/>
                </Container>

            </Container>
        </div>
    );
}

export default Predictions;