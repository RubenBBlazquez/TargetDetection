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
import {Container, Form} from "react-bootstrap";

const data = [
    {
        name: 'Pred A',
        fail: 4000,
        success: 2400,
        amt: 2400,
    },
    {
        name: 'Pred B',
        fail: 3000,
        success: 1398,
        amt: 2210,
    },
    {
        name: 'Pred C',
        fail: 2000,
        success: 9800,
        amt: 2290,
    },
    {
        name: 'Pred D',
        fail: 2780,
        success: 3908,
        amt: 2000,
    },
    {
        name: 'Pred E',
        fail: 1890,
        success: 4800,
        amt: 2181,
    },
    {
        name: 'Pred F',
        fail: 2390,
        success: 3800,
        amt: 2500,
    },
    {
        name: 'Pred G',
        fail: 3490,
        success: 4300,
        amt: 2100,
    },
];

export default class MisfireOfLastPredictionsChart extends PureComponent {
    constructor(props) {
        super(props);
        this.state = {
            useBarChart: true
        }

        this.setMisfireBarChart = this.setMisfireBarChart.bind(this);
    }

    setMisfireBarChart = (useBarChart) => {
        this.setState({useBarChart});
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
                    <Bar dataKey="success" fill="#82ca9d"/>
                    <Bar dataKey="fail" fill="red"/>
                </BarChart>
            </ResponsiveContainer>
        );

        if (!useBarChart) {
            chart =  (
                <ResponsiveContainer width='100%' height={330}>
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
                        <Line type="monotone" dataKey="success" stroke="#CD5C5C" activeDot={{r: 8}}/>
                        <Line type="monotone" dataKey="fail" stroke="#008080"/>
                    </LineChart>
                </ResponsiveContainer>
            );
        }

        return (
            <Container className={'col-lg-4 col-xs-12 mt-lg-0 mt-sm-3 border rounded'}>
                <div className={'d-flex flex-column align-items-center'}>
                    <h4 className={'text-center'}>Misfire of last predictions</h4>
                    <div className={'w-100 p-2'}>
                        <Form.Check
                            type="switch"
                            label="Change to Linear Chart"
                            id="set-linear-chart-misfires"
                            className={`float-end ${!useBarChart && 'd-none'}`}
                            checked={false}
                            onChange={() => {
                                this.setMisfireBarChart(false);
                            }}
                        />
                        <Form.Check
                            type="switch"
                            label="Change to Bar Chart"
                            id="set-bar-chart-misfires"
                            className={`float-end ${useBarChart && 'd-none'}`}
                            checked={true}
                            onChange={() => {
                                this.setMisfireBarChart(true);
                            }}
                        />
                    </div>
                </div>
                {
                    chart
                }
            </Container>
        )
    }
}
