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
    render() {
        const {useLineChart} = this.props;
        console.log(useLineChart)
        if (useLineChart) {
            return (
                <ResponsiveContainer  width='100%' height={300}>
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
                        <Line type="monotone" dataKey="pred_0" stroke="#8884d8" activeDot={{r: 8}}/>
                        <Line type="monotone" dataKey="pred_1" stroke="#82ca9d"/>
                    </LineChart>
                </ResponsiveContainer>
            );
        }

        return (
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
                    <Bar dataKey="pred_0" fill="#8884d8"/>
                    <Bar dataKey="pred_1" fill="#82ca9d"/>
                </BarChart>
            </ResponsiveContainer>
        );
    }
}
