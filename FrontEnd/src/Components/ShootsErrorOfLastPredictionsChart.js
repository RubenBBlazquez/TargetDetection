import React, {PureComponent} from 'react';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    LineChart,
    Line
} from 'recharts';
import PredictionsByDateChart from "./PredictionsByDateChart";
import {Container, Form} from "react-bootstrap";

const data = [
    {
        name: 'Pred A',
        first: 4000,
        second: 2400,
        third: 2400,
        amt: 2400,
    },
    {
        name: 'Pred B',
        first: 3000,
        second: 1398,
        third: 1400,
        amt: 2210,
    },
    {
        name: 'Pred C',
        first: 2000,
        second: 9800,
        third: 9900,
        amt: 2290,
    },
    {
        name: 'Pred D',
        first: 2780,
        second: 3908,
        third: 4008,
        amt: 2000,
    },
    {
        name: 'Pred E',
        first: 1890,
        second: 4800,
        third: 4900,
        amt: 2181,
    },
    {
        name: 'Pred F',
        first: 2390,
        second: 3800,
        third: 3900,
        amt: 2500,
    },
    {
        name: 'Pred G',
        first: 3490,
        second: 4300,
        third: 4400,
        amt: 2100,
    },
];

export default class MisfireOfLastPredictionsChart extends PureComponent {
    constructor(props) {
        super(props);
        this.state = {
            useBarChart: true
        }
    }

    /**
     * Set the chart to use bar chart or line chart
     *
     * @param {boolean} setBarChart
     */
    setBarChart = (setBarChart) => {
        this.setState({
            useBarChart: setBarChart
        })
    }

    render() {
        const {useBarChart} = this.state;

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
                    <Bar dataKey="first" fill="#CD5C5C"/>
                    <Bar dataKey="second" fill="#008080"/>
                    <Bar dataKey="third" fill="#F5B041"/>
                </BarChart>
            </ResponsiveContainer>
        );

        if (!useBarChart) {
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
                        <Line type="monotone" dataKey="first" stroke="#CD5C5C" activeDot={{r: 8}}/>
                        <Line type="monotone" dataKey="second" stroke="#008080"/>
                        <Line type="monotone" dataKey="third" stroke="#F5B041"/>
                    </LineChart>
                </ResponsiveContainer>
            );
        }

        return (
            <Container className={'col-lg-4 col-xs-12 mt-lg-0 mt-sm-3 border rounded'}>
                <div className={'d-flex flex-column align-items-center'}>
                    <h4 className={'text-center'}>Shoots Error of last predictions</h4>
                    <div className={'w-100 p-2'}>
                        <Form.Check
                            type="switch"
                            label="Change to Linear Chart"
                            id="set-linear-chart-shoots-error"
                            className={`float-end ${!useBarChart && 'd-none'}`}
                            checked={false}
                            onChange={() => {
                                this.setBarChart(false);
                            }}
                        />
                        <Form.Check
                            type="switch"
                            label="Change to Bar Chart"
                            id="set-bar-chart-shoots-error"
                            className={`float-end ${useBarChart && 'd-none'}`}
                            checked={true}
                            onChange={() => {
                                this.setBarChart(true);
                            }}
                        />
                    </div>
                </div>
                {chart}
            </Container>
        )
    }
}
