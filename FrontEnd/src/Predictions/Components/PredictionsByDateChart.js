import React from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    BarChart,
    Bar
} from 'recharts';
import {Container, Form} from "react-bootstrap";

const data = [
    {
        name: 'Date A',
        pred_1: 4000,
        pred_0: 2400,
        amt: 2400,
    },
    {
        name: 'Date B',
        pred_1: 3000,
        pred_0: 1398,
        amt: 2210,
    },
    {
        name: 'Date C',
        pred_1: 2000,
        pred_0: 9800,
        amt: 2290,
    },
    {
        name: 'Date D',
        pred_1: 2780,
        pred_0: 3908,
        amt: 2000,
    },
    {
        name: 'Date E',
        pred_1: 1890,
        pred_0: 4800,
        amt: 2181,
    },
    {
        name: 'Date F',
        pred_1: 2390,
        pred_0: 3800,
        amt: 2500,
    },
    {
        name: 'Date G',
        pred_1: 3490,
        pred_0: 4300,
        amt: 2100,
    },
];

export default class PredictionsByDateChart extends React.PureComponent {
    constructor(props) {
        super(props);
        this.state = {
            useLineChart: true,
        }

        this.manageBarClick.bind(this);
        this.manageLineClick.bind(this);
    }

    manageBarClick = (data) => {
        console.log(data)
    }
    manageLineClick = (event, data) => {
        console.log(data)
    }

    /**
     * Method to set the state of the linearChartByPD
     *
     * @param {boolean} setLinearChart
     */
    setLinearChartByPD = (setLinearChart) => {
        this.setState({useLineChart: setLinearChart})
    }

    render() {
        const {useLineChart} = this.state;

        let chart = (
            <ResponsiveContainer width="100%" height={300}>
                <BarChart
                    width={500}
                    height={300}
                    data={data}
                    margin={{
                        top: 5,
                        right: 30,
                        left: 20,
                        bottom: 5,
                    }}
                >
                    <CartesianGrid strokeDasharray="3 3"/>
                    <XAxis dataKey="name"/>
                    <YAxis/>
                    <Tooltip/>
                    <Legend/>
                    <Bar onClick={this.manageBarClick} dataKey="pred_0" fill="#8884d8"/>
                    <Bar onClick={this.manageBarClick} dataKey="pred_1" fill="#82ca9d"/>
                </BarChart>
            </ResponsiveContainer>
        );

        if (useLineChart) {
            chart = (
                <ResponsiveContainer width='100%' height={300}>
                    <LineChart
                        width={500}
                        height={300}
                        data={data}
                        margin={{
                            top: 5,
                            right: 30,
                            left: 20,
                            bottom: 5,
                        }}
                    >
                        <CartesianGrid strokeDasharray="3 3"/>
                        <XAxis dataKey="name"/>
                        <YAxis/>
                        <Tooltip/>
                        <Legend
                        />
                        <Line type="monotone" dataKey="pred_0" stroke="#8884d8"
                              activeDot={{onClick: this.manageLineClick}}/>
                        <Line type="monotone" dataKey="pred_1" stroke="#82ca9d"
                              activeDot={{onClick: this.manageLineClick}}/>
                    </LineChart>
                </ResponsiveContainer>
            );
        }

        return (
            <Container className={'col-lg-4 col-sm-12 mt-lg-0 mt-sm-3 border rounded'}>
                <div className={'d-flex flex-column align-items-center'}>
                    <h4 className={'text-center'}>Predictions By Date</h4>
                    <div className={'w-100 p-2'}>
                        <Form.Check
                            type="switch"
                            label="Change to Linear Chart"
                            id="set-linear-chart-predictions-by-date"
                            className={`float-end ${useLineChart && 'd-none'}`}
                            checked={true}
                            onChange={() => {
                                this.setLinearChartByPD(true);
                            }}
                        />
                        <Form.Check
                            type="switch"
                            label="Change to Bar Chart"
                            id="set-bar-chart-predictions-by-date"
                            checked={false}
                            className={`float-end ${!useLineChart && 'd-none'}`}
                            onChange={() => {
                                this.setLinearChartByPD(false);
                            }}
                        />
                    </div>
                </div>
                {
                    chart
                }
            </Container>
        );
    }
}