// Global variables
let metricsChart = null;
let channelAnalysisChart24 = null;
let channelAnalysisChart5 = null;
let currentPoint = null;

// DOM Elements
const pointSelect = document.getElementById('point-select');
const metricsChartCanvas = document.getElementById('predictions-chart');
const channelAnalysisCanvas24 = document.getElementById('channel-analysis-chart-2.4');
const channelAnalysisCanvas5 = document.getElementById('channel-analysis-chart-5');
const channelPerformanceTable24 = document.getElementById('channel-performance-table-2.4');
const channelPerformanceTable5 = document.getElementById('channel-performance-table-5');
const dashboardTab = document.getElementById('dashboard-tab');
const pointsTab = document.getElementById('points-tab');
const dashboardContent = document.getElementById('dashboard-content');
const pointsContent = document.getElementById('points-content');

// Helper function to convert minutes to HH:MM format
function minutesToTime(minutes) {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}`;
}

// Initialize the application
async function init() {
    await loadPoints();
    await loadSummary();
    setupEventListeners();
}

// Load available points
async function loadPoints() {
    try {
        const response = await fetch('/api/points');
        const points = await response.json();
        
        // Clear existing options
        pointSelect.innerHTML = '<option value="">Выберите точку...</option>';
        
        // Add points to select
        points.forEach(point => {
            const option = document.createElement('option');
            option.value = point.id;
            option.textContent = `${point.name} (${point.band})`;
            pointSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Ошибка загрузки точек:', error);
    }
}

// Load summary data
async function loadSummary() {
    try {
        const response = await fetch('/api/summary');
        const summary = await response.json();
        updateSummaryDisplay(summary);
    } catch (error) {
        console.error('Ошибка загрузки сводки:', error);
    }
}

// Update summary display
function updateSummaryDisplay(summary) {
    const summaryContainer = document.getElementById('summary-content');
    if (!summaryContainer) return;

    summaryContainer.innerHTML = `
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div class="bg-white p-4 rounded-lg shadow">
                <h3 class="text-lg font-semibold mb-2 text-gray-900">Всего точек</h3>
                <p class="text-2xl font-bold text-gray-900">${summary.total_points}</p>
            </div>
            <div class="bg-white p-4 rounded-lg shadow">
                <h3 class="text-lg font-semibold mb-2 text-gray-900">Конфликты</h3>
                <p class="text-2xl font-bold text-gray-900">${summary.total_conflicts}</p>
            </div>
            <div class="bg-white p-4 rounded-lg shadow">
                <h3 class="text-lg font-semibold mb-2 text-gray-900 break-words">Распределение частот</h3>
                <div class="space-y-1">
                    <p class="text-sm text-gray-900">2.4 ГГц: ${summary.bands['2.4 GHz']}</p>
                    <p class="text-sm text-gray-900">5 ГГц: ${summary.bands['5 GHz']}</p>
                </div>
            </div>
        </div>
    `;
}

// Load metrics for a point
async function loadPointMetrics(pointId) {
    try {
        const response = await fetch(`/api/points/${pointId}/metrics`);
        const data = await response.json();
        
        // Split data by frequency band
        const data24 = {
            time_periods: data.time_periods.filter(p => p.band === '2.4 GHz')
        };
        const data5 = {
            time_periods: data.time_periods.filter(p => p.band === '5 GHz')
        };
        
        updateChannelAnalysis(data24, data5);
        updateChannelPerformanceTable(data24, data5);
    } catch (error) {
        console.error('Ошибка загрузки метрик:', error);
    }
}

// Update the channel analysis charts
function updateChannelAnalysis(data24, data5) {
    // Update 2.4 GHz chart
    updateSingleChart(data24, channelAnalysisCanvas24, channelAnalysisChart24, '2.4');
    // Update 5 GHz chart
    updateSingleChart(data5, channelAnalysisCanvas5, channelAnalysisChart5, '5');
}

function updateSingleChart(data, canvas, chart, band) {
    const timePeriods = data.time_periods;
    
    // Prepare data for Chart.js
    const timestamps = timePeriods.map(p => minutesToTime(p.start_time));
    const channels = timePeriods.map(p => p.channel);
    
    // Destroy existing chart if it exists
    if (chart) {
        chart.destroy();
    }
    
    // Create new chart
    chart = new Chart(canvas, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [
                {
                    label: 'Канал',
                    data: channels,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    title: {
                        display: true,
                        text: 'Канал'
                    }
                }
            }
        }
    });
    
    // Update the global chart reference
    if (band === '2.4') {
        channelAnalysisChart24 = chart;
    } else {
        channelAnalysisChart5 = chart;
    }
}

// Update the channel performance tables
function updateChannelPerformanceTable(data24, data5) {
    updateSingleTable(data24, channelPerformanceTable24);
    updateSingleTable(data5, channelPerformanceTable5);
}

function updateSingleTable(data, table) {
    const timePeriods = data.time_periods;
    
    // Clear existing rows
    table.innerHTML = '';
    
    // Add new rows
    timePeriods.forEach(period => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                ${minutesToTime(period.start_time)} - ${minutesToTime(period.end_time)}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                ${period.channel}
            </td>
        `;
        table.appendChild(row);
    });
}

// Setup event listeners
function setupEventListeners() {
    // Point selection change
    pointSelect.addEventListener('change', (e) => {
        const pointId = e.target.value;
        if (pointId) {
            currentPoint = pointId;
            loadPointMetrics(pointId);
        }
    });
    
    // Tab navigation
    dashboardTab.addEventListener('click', (e) => {
        e.preventDefault();
        showTab('dashboard');
    });
    
    pointsTab.addEventListener('click', (e) => {
        e.preventDefault();
        showTab('points');
    });
}

// Show selected tab
function showTab(tabName) {
    // Hide all content
    dashboardContent.classList.add('hidden');
    pointsContent.classList.add('hidden');
    
    // Remove active state from all tabs
    dashboardTab.classList.remove('border-indigo-500', 'text-gray-900');
    dashboardTab.classList.add('border-transparent', 'text-gray-500');
    pointsTab.classList.remove('border-indigo-500', 'text-gray-900');
    pointsTab.classList.add('border-transparent', 'text-gray-500');
    
    // Show selected content and update tab
    switch (tabName) {
        case 'dashboard':
            dashboardContent.classList.remove('hidden');
            dashboardTab.classList.remove('border-transparent', 'text-gray-500');
            dashboardTab.classList.add('border-indigo-500', 'text-gray-900');
            break;
        case 'points':
            pointsContent.classList.remove('hidden');
            pointsTab.classList.remove('border-transparent', 'text-gray-500');
            pointsTab.classList.add('border-indigo-500', 'text-gray-900');
            break;
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', init); 