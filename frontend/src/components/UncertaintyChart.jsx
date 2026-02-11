
import React from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler } from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

const UncertaintyChart = ({ data }) => {
    if (!data || data.length === 0) return <div className="flex h-full items-center justify-center text-gray-400">No data available</div>;

    const chartData = {
        labels: data.map((_, i) => `Q${i + 1}`),
        datasets: [
            {
                label: 'Semantic Entropy (Uncertainty)',
                data: data.map(d => d.entropy),
                borderColor: '#ff3b30',
                backgroundColor: 'rgba(255, 59, 48, 0.1)',
                tension: 0.4,
                fill: true,
            },
            {
                label: 'Accuracy Probability',
                data: data.map(d => d.accuracy),
                borderColor: '#0071e3',
                backgroundColor: 'rgba(0, 113, 227, 0.05)',
                tension: 0.4,
                fill: true,
            }
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
                align: 'end',
                labels: {
                    boxWidth: 10,
                    usePointStyle: true,
                    font: { family: 'Inter', size: 11 }
                }
            },
            tooltip: {
                mode: 'index',
                intersect: false,
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                titleColor: '#000',
                bodyColor: '#666',
                borderColor: '#ddd',
                borderWidth: 1,
                padding: 10,
                displayColors: true,
                callbacks: {
                    title: (items) => {
                        const idx = items[0].dataIndex;
                        return data[idx].question;
                    }
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 1.0,
                grid: {
                    color: '#f0f0f0'
                },
                ticks: {
                    font: { family: 'Inter', size: 10 }
                }
            },
            x: {
                grid: {
                    display: false
                },
                ticks: {
                    font: { family: 'Inter', size: 10 }
                }
            }
        },
        interaction: {
            mode: 'nearest',
            axis: 'x',
            intersect: false
        }
    };

    return <Line data={chartData} options={options} />;
};

export default UncertaintyChart;
