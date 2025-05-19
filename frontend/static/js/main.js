// Global variables
let predictionsChart = null;
let currentPoint = null;

// DOM Elements
const pointSelect = document.getElementById('point-select');
const predictionsChartCanvas = document.getElementById('predictions-chart');
const recommendationsTable = document.getElementById('recommendations-table');
const dashboardTab = document.getElementById('dashboard-tab');
const settingsTab = document.getElementById('settings-tab');
const pointsTab = document.getElementById('points-tab');
const dashboardContent = document.getElementById('dashboard-content');
const settingsContent = document.getElementById('settings-content');
const pointsContent = document.getElementById('points-content');

// Initialize the application
async function init() {
    await loadPoints();
    setupEventListeners();
}

// Load available points
async function loadPoints() {
    try {
        const response = await fetch('/api/points');
        const points = await response.json();
        
        // Clear existing options
        pointSelect.innerHTML = '<option value="">Select a point...</option>';
        
        // Add points to select
        points.forEach(point => {
            const option = document.createElement('option');
            option.value = point.id;
            option.textContent = `${point.name} (${point.type})`;
            pointSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading points:', error);
    }
}

// Load predictions for a point
async function loadPredictions(pointId) {
    try {
        const response = await fetch(`/api/predictions?point_id=${pointId}`);
        const data = await response.json();
        updateChart(data);
        updateRecommendationsTable(data);
    } catch (error) {
        console.error('Error loading predictions:', error);
    }
}

// Update the predictions chart
function updateChart(data) {
    const predictions = data.predictions;
    
    // Prepare data for Chart.js
    const timestamps = predictions.map(p => new Date(p.timestamp));
    const channels = predictions.map(p => p.channel);
    const loads = predictions.map(p => p.load);
    
    // Destroy existing chart if it exists
    if (predictionsChart) {
        predictionsChart.destroy();
    }
    
    // Create new chart
    predictionsChart = new Chart(predictionsChartCanvas, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [
                {
                    label: 'Channel',
                    data: channels,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1,
                    yAxisID: 'y'
                },
                {
                    label: 'Load',
                    data: loads,
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1,
                    yAxisID: 'y1'
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
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Channel'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Load'
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
}

// Update the recommendations table
function updateRecommendationsTable(data) {
    const predictions = data.predictions;
    
    // Clear existing rows
    recommendationsTable.innerHTML = '';
    
    // Add new rows
    predictions.forEach((prediction, index) => {
        const row = document.createElement('tr');
        
        // Get current and next channel
        const currentChannel = index > 0 ? predictions[index - 1].channel : prediction.channel;
        const nextChannel = prediction.channel;
        
        row.innerHTML = `
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                ${new Date(prediction.timestamp).toLocaleString()}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                ${currentChannel}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                ${nextChannel}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                ${(prediction.load * 100).toFixed(1)}%
            </td>
        `;
        
        recommendationsTable.appendChild(row);
    });
}

// Setup event listeners
function setupEventListeners() {
    // Point selection change
    pointSelect.addEventListener('change', (e) => {
        const pointId = e.target.value;
        if (pointId) {
            currentPoint = pointId;
            loadPredictions(pointId);
        }
    });
    
    // Tab navigation
    dashboardTab.addEventListener('click', (e) => {
        e.preventDefault();
        showTab('dashboard');
    });
    
    settingsTab.addEventListener('click', (e) => {
        e.preventDefault();
        showTab('settings');
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
    settingsContent.classList.add('hidden');
    pointsContent.classList.add('hidden');
    
    // Remove active state from all tabs
    dashboardTab.classList.remove('border-indigo-500', 'text-gray-900');
    dashboardTab.classList.add('border-transparent', 'text-gray-500');
    settingsTab.classList.remove('border-indigo-500', 'text-gray-900');
    settingsTab.classList.add('border-transparent', 'text-gray-500');
    pointsTab.classList.remove('border-indigo-500', 'text-gray-900');
    pointsTab.classList.add('border-transparent', 'text-gray-500');
    
    // Show selected content and update tab
    switch (tabName) {
        case 'dashboard':
            dashboardContent.classList.remove('hidden');
            dashboardTab.classList.remove('border-transparent', 'text-gray-500');
            dashboardTab.classList.add('border-indigo-500', 'text-gray-900');
            break;
        case 'settings':
            settingsContent.classList.remove('hidden');
            settingsTab.classList.remove('border-transparent', 'text-gray-500');
            settingsTab.classList.add('border-indigo-500', 'text-gray-900');
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