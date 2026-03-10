
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
                label: 'Confidence',
                data: data.map(d => d.confidence ?? 0),
                borderColor: '#0071e3',
                backgroundColor: (ctx) => {
                    const chart = ctx.chart;
                    const { ctx: canvasCtx, chartArea } = chart;
                    if (!chartArea) return 'rgba(0, 113, 227, 0.1)';
                    const gradient = canvasCtx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
                    gradient.addColorStop(0, 'rgba(255, 59, 48, 0.15)');    // Red at bottom (low confidence)
                    gradient.addColorStop(0.5, 'rgba(255, 179, 0, 0.1)');  // Amber in middle
                    gradient.addColorStop(1, 'rgba(52, 199, 89, 0.15)');    // Green at top (high confidence)
                    return gradient;
                },
                tension: 0.4,
                fill: true,
                pointBackgroundColor: (ctx) => {
                    const val = ctx.raw;
                    if (val >= 0.75) return '#34c759';
                    if (val >= 0.5) return '#ffb300';
                    return '#ff3b30';
                },
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 5,
                pointHoverRadius: 7,
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
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                titleColor: '#000',
                bodyColor: '#666',
                borderColor: '#ddd',
                borderWidth: 1,
                padding: 12,
                displayColors: false,
                callbacks: {
                    title: (items) => {
                        const idx = items[0].dataIndex;
                        return data[idx].question;
                    },
                    label: (item) => {
                        const val = item.raw;
                        const pct = (val * 100).toFixed(1);
                        const label = val >= 0.75 ? 'Reliable' : val >= 0.5 ? 'Uncertain' : 'Hallucinated';
                        return `Confidence: ${pct}% (${label})`;
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
                    font: { family: 'Inter', size: 10 },
                    callback: (v) => `${(v * 100).toFixed(0)}%`
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
