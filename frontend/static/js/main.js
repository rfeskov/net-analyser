// Global variables
let metricsChart = null;
let channelAnalysisChart24 = null;
let channelAnalysisChart5 = null;
let channelLoadChart24 = null;
let channelLoadChart5 = null;
let channelSignalChart24 = null;
let channelSignalChart5 = null;
let currentPoint = null;

// DOM Elements
const pointSelect = document.getElementById('point-select');
const metricsChartCanvas = document.getElementById('predictions-chart');
const channelAnalysisCanvas24 = document.getElementById('channel-analysis-chart-2.4');
const channelAnalysisCanvas5 = document.getElementById('channel-analysis-chart-5');
const channelLoadCanvas24 = document.getElementById('channel-load-chart-2.4');
const channelLoadCanvas5 = document.getElementById('channel-load-chart-5');
const channelSignalCanvas24 = document.getElementById('channel-signal-chart-2.4');
const channelSignalCanvas5 = document.getElementById('channel-signal-chart-5');
const channelPerformanceTable24 = document.getElementById('channel-performance-table-2.4');
const channelPerformanceTable5 = document.getElementById('channel-performance-table-5');

// Фиксированные цвета для каналов
const CHANNEL_COLORS = {
    '1': '#FF5733',  // Красный
    '2': '#33FF57',  // Зеленый
    '3': '#3357FF',  // Синий
    '4': '#F3FF33',  // Желтый
    '5': '#FF33F3',  // Розовый
    '6': '#33FFF3',  // Голубой
    '7': '#F333FF',  // Фиолетовый
    '8': '#FF8333',  // Оранжевый
    '9': '#33FF83',  // Светло-зеленый
    '10': '#8333FF', // Темно-фиолетовый
    '11': '#FF3333', // Ярко-красный
    '12': '#33FF33', // Ярко-зеленый
    '13': '#3333FF', // Ярко-синий
    '36': '#FF5733',
    '40': '#33FF57',
    '44': '#3357FF',
    '48': '#F3FF33',
    '52': '#FF33F3',
    '56': '#33FFF3',
    '60': '#F333FF',
    '64': '#FF8333',
    '100': '#33FF83',
    '104': '#8333FF',
    '108': '#FF3333',
    '112': '#33FF33',
    '116': '#3333FF',
    '120': '#FF5733',
    '124': '#33FF57',
    '128': '#3357FF',
    '132': '#F3FF33',
    '136': '#FF33F3',
    '140': '#33FFF3',
    '144': '#F333FF',
    '149': '#FF8333',
    '153': '#33FF83',
    '157': '#8333FF',
    '161': '#FF3333',
    '165': '#33FF33'
};

// Helper function to convert minutes to HH:MM format
function minutesToTime(minutes) {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}`;
}

// Helper function to generate random color
function getRandomColor() {
    const letters = '0123456789ABCDEF';
    let color = '#';
    for (let i = 0; i < 6; i++) {
        color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
}

// Initialize the application
async function init() {
    await loadPoints();
    await loadSummary();
    setupEventListeners();
    showTab('dashboard'); // Show dashboard tab by default
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

        // Update points grid
        updatePointsGrid(points);
    } catch (error) {
        console.error('Ошибка загрузки точек:', error);
    }
}

// Update points grid with cards
function updatePointsGrid(points) {
    const grid = document.getElementById('points-grid');
    if (!grid) return;

    grid.innerHTML = '';

    points.forEach(point => {
        const card = document.createElement('div');
        card.className = 'bg-white rounded-lg shadow p-6 hover:shadow-lg transition-shadow duration-200';
        
        // Determine status color
        const statusColor = point.is_online ? 'bg-green-500' : 'bg-red-500';
        
        card.innerHTML = `
            <div class="flex items-start justify-between mb-4">
                <div>
                    <h3 class="text-lg font-semibold text-gray-900">${point.name}</h3>
                    <p class="text-sm text-gray-500">${point.band}</p>
                </div>
                <div class="flex items-center">
                    <span class="inline-block w-3 h-3 rounded-full ${statusColor} mr-2"></span>
                    <span class="text-sm text-gray-600">${point.is_online ? 'Онлайн' : 'Оффлайн'}</span>
                </div>
            </div>
            <div class="space-y-2">
                <div class="flex justify-between text-sm">
                    <span class="text-gray-500">Клиентов:</span>
                    <span class="text-gray-900 font-medium">${point.clients_count || 0}</span>
                </div>
                <div class="flex justify-between text-sm">
                    <span class="text-gray-500">Канал:</span>
                    <span class="text-gray-900 font-medium">${point.channel || 'Н/Д'}</span>
                </div>
                <div class="flex justify-between text-sm">
                    <span class="text-gray-500">Уровень сигнала:</span>
                    <span class="text-gray-900 font-medium">${point.signal_strength ? point.signal_strength + ' dBm' : 'Н/Д'}</span>
                </div>
            </div>
            <div class="mt-4 pt-4 border-t border-gray-100">
                <button 
                    class="w-full bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 transition-colors duration-200"
                    onclick="selectPoint('${point.id}')"
                >
                    Подробнее
                </button>
            </div>
        `;
        
        grid.appendChild(card);
    });
}

// Select point from card
function selectPoint(pointId) {
    pointSelect.value = pointId;
    loadPointMetrics(pointId);
    showTab('points');
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

    // Get current date and time
    const now = new Date();
    const dateStr = now.toLocaleDateString('ru-RU', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
    const timeStr = now.toLocaleTimeString('ru-RU', {
        hour: '2-digit',
        minute: '2-digit'
    });

    // Calculate total clients
    const totalClients = 56;
    //const totalClients = summary.bands['2.4 GHz'] + summary.bands['5 GHz'];
    // Используем одинаковое количество точек для обеих частот
    const totalPoints = summary.total_points;

    summaryContainer.innerHTML = `
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            <div class="bg-white p-6 rounded-lg shadow">
                <h3 class="text-xl font-semibold mb-4 text-gray-900">Сеть</h3>
                <div>
                    <p class="text-base text-gray-600 mb-2">Название: Office-Network</p>
                    <p class="text-base text-gray-600">Подразделение: Главный офис</p>
                </div>
            </div>
            <div class="bg-white p-6 rounded-lg shadow">
                <h3 class="text-xl font-semibold mb-4 text-gray-900">Статистика</h3>
                <div>
                    <p class="text-base text-gray-600 mb-2">Всего точек: ${totalPoints}</p>
                    <p class="text-base text-gray-600">Всего клиентов: ${totalClients}</p>
                </div>
            </div>
            <div class="bg-white p-6 rounded-lg shadow">
                <h3 class="text-xl font-semibold mb-4 text-gray-900">Распределение частот</h3>
                <div>
                    <p class="text-base text-gray-600 mb-2">2.4 ГГц: ${totalPoints} точек</p>
                    <p class="text-base text-gray-600">5 ГГц: ${totalPoints} точек</p>
                </div>
            </div>
            <div class="bg-white p-6 rounded-lg shadow">
                <h3 class="text-xl font-semibold mb-4 text-gray-900">Сегодня</h3>
                <div>
                    <p class="text-base text-gray-600 mb-2">Дата: ${dateStr}</p>
                    <p class="text-base text-gray-600">Время: ${timeStr}</p>
                </div>
            </div>
        </div>
    `;
}

// Load metrics for a point
async function loadPointMetrics(pointId) {
    try {
        const [metricsResponse, loadResponse] = await Promise.all([
            fetch(`/api/points/${pointId}/metrics`),
            fetch(`/api/points/${pointId}/channel_load`)
        ]);
        
        const metricsData = await metricsResponse.json();
        const loadData = await loadResponse.json();
        
        // Split metrics data by frequency band
        const data24 = {
            time_periods: metricsData.time_periods.filter(p => p.band === '2.4 GHz')
        };
        const data5 = {
            time_periods: metricsData.time_periods.filter(p => p.band === '5 GHz')
        };
        
        updateChannelAnalysis(data24, data5);
        updateChannelPerformanceTable(data24, data5);
        updateChannelLoad(loadData);
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

// Update channel load charts
function updateChannelLoad(data) {
    // Update 2.4 GHz charts
    updateSingleLoadChart(data['2.4 GHz'], channelLoadCanvas24, channelLoadChart24, '2.4');
    updateSingleSignalChart(data['2.4 GHz'], channelSignalCanvas24, channelSignalChart24, '2.4');
    // Update 5 GHz charts
    updateSingleLoadChart(data['5 GHz'], channelLoadCanvas5, channelLoadChart5, '5');
    updateSingleSignalChart(data['5 GHz'], channelSignalCanvas5, channelSignalChart5, '5');
}

function updateSingleLoadChart(bandData, canvas, chart, band) {
    if (!bandData) return;
    
    // Destroy existing chart if it exists
    if (chart) {
        chart.destroy();
    }
    
    // Создаем контейнер для графика и чекбоксов
    const container = canvas.parentElement;
    
    // Удаляем только существующий wrapper, если он есть
    const existingWrapper = container.querySelector('.flex.gap-8');
    if (existingWrapper) {
        existingWrapper.remove();
    }
    
    // Восстанавливаем оригинальный canvas
    container.appendChild(canvas);
    
    const wrapper = document.createElement('div');
    wrapper.className = 'flex gap-8';
    
    // Создаем контейнер для графика
    const chartContainer = document.createElement('div');
    chartContainer.className = 'flex-grow';
    chartContainer.style.minHeight = '300px';
    chartContainer.appendChild(canvas);
    
    // Создаем контейнер для чекбоксов
    const toggleContainer = document.createElement('div');
    toggleContainer.className = 'channel-toggles';
    toggleContainer.style.width = '150px';
    toggleContainer.style.maxHeight = '300px';
    toggleContainer.style.overflowY = 'auto';
    toggleContainer.style.flexShrink = '0';
    
    // Создаем массивы для часов (0-23)
    const hours = Array.from({length: 24}, (_, i) => i);
    const hourLabels = hours.map(h => `${h.toString().padStart(2, '0')}:00`);
    
    // Prepare datasets for each channel
    const datasets = [];
    const channelStates = {};
    
    Object.entries(bandData).forEach(([channel, data]) => {
        // Создаем чекбокс для канала
        const label = document.createElement('label');
        label.className = 'flex items-center space-x-2 text-sm mb-2 whitespace-nowrap';
        label.style.color = CHANNEL_COLORS[channel] || '#000000';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = true;
        checkbox.className = 'channel-toggle';
        checkbox.dataset.channel = channel;
        
        const span = document.createElement('span');
        span.textContent = `Канал ${channel}`;
        
        label.appendChild(checkbox);
        label.appendChild(span);
        toggleContainer.appendChild(label);
        
        // Сохраняем состояние канала
        channelStates[channel] = true;
        
        // Группируем данные по часам
        const hourlyClients = new Array(24).fill(0);
        const hourlyCount = new Array(24).fill(0);
        
        data.times.forEach((time, index) => {
            const hour = Math.floor(time / 60);
            if (hour >= 0 && hour < 24) {
                hourlyClients[hour] += parseInt(data.clients[index]) || 0;
                hourlyCount[hour]++;
            }
        });
        
        // Вычисляем средние значения для каждого часа
        const averageClients = hourlyClients.map((sum, i) => 
            hourlyCount[i] > 0 ? Math.round(sum / hourlyCount[i]) : 0
        );
        
        datasets.push({
            label: `Канал ${channel}`,
            data: averageClients,
            backgroundColor: CHANNEL_COLORS[channel] || '#000000',
            borderColor: 'white',
            borderWidth: 1,
            hidden: false
        });
    });
    
    // Собираем контейнер
    wrapper.appendChild(chartContainer);
    wrapper.appendChild(toggleContainer);
    
    // Добавляем новый контейнер
    container.appendChild(wrapper);
    
    // Create new chart
    chart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: hourLabels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            layout: {
                padding: {
                    top: 30,
                    bottom: 30
                }
            },
            scales: {
                x: {
                    stacked: true,
                    title: {
                        display: true,
                        text: 'Время'
                    },
                    grid: {
                        display: true,
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                y: {
                    stacked: true,
                    title: {
                        display: true,
                        text: 'Количество клиентов'
                    },
                    min: 0,
                    grid: {
                        display: true,
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y} клиентов`;
                        }
                    }
                },
                legend: {
                    display: false
                }
            }
        }
    });
    
    // Добавляем обработчики событий для чекбоксов
    toggleContainer.querySelectorAll('.channel-toggle').forEach(checkbox => {
        checkbox.addEventListener('change', (e) => {
            const channel = e.target.dataset.channel;
            const datasetIndex = datasets.findIndex(ds => ds.label === `Канал ${channel}`);
            if (datasetIndex !== -1) {
                chart.data.datasets[datasetIndex].hidden = !e.target.checked;
                chart.update();
            }
        });
    });
    
    // Update the global chart reference
    if (band === '2.4') {
        channelLoadChart24 = chart;
    } else {
        channelLoadChart5 = chart;
    }
}

function updateSingleSignalChart(bandData, canvas, chart, band) {
    if (!bandData) return;
    
    // Destroy existing chart if it exists
    if (chart) {
        chart.destroy();
    }
    
    // Создаем массивы для часов (0-23)
    const hours = Array.from({length: 24}, (_, i) => i);
    const hourLabels = hours.map(h => `${h.toString().padStart(2, '0')}:00`);
    
    // Prepare datasets for each channel
    const datasets = [];
    
    Object.entries(bandData).forEach(([channel, data]) => {
        // Группируем данные по часам
        const hourlySignals = new Array(24).fill(0);
        const hourlyCount = new Array(24).fill(0);
        
        data.times.forEach((time, index) => {
            const hour = Math.floor(time / 60);
            if (hour >= 0 && hour < 24) {
                hourlySignals[hour] += parseFloat(data.signal[index]) || 0;
                hourlyCount[hour]++;
            }
        });
        
        // Вычисляем средние значения для каждого часа
        const averageSignals = hourlySignals.map((sum, i) => {
            if (hourlyCount[i] === 0) return null; // Возвращаем null для часов без данных
            const avg = sum / hourlyCount[i];
            return avg === 0 ? null : avg.toFixed(1); // Возвращаем null для нулевых значений
        });
        
        datasets.push({
            label: `Канал ${channel}`,
            data: averageSignals,
            borderColor: CHANNEL_COLORS[channel] || '#000000',
            backgroundColor: 'transparent',
            borderWidth: 2,
            tension: 0.1,
            hidden: false,
            spanGaps: true // Позволяет линиям проходить через пропуски
        });
    });
    
    // Create new chart
    chart = new Chart(canvas, {
        type: 'line',
        data: {
            labels: hourLabels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Время'
                    },
                    grid: {
                        display: true,
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Уровень помех (dBm)'
                    },
                    min: -100,
                    max: 0,
                    grid: {
                        display: true,
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.parsed.y === null) return null;
                            return `${context.dataset.label}: ${context.parsed.y} dBm`;
                        }
                    }
                },
                legend: {
                    display: false
                }
            }
        }
    });
    
    // Update the global chart reference
    if (band === '2.4') {
        channelSignalChart24 = chart;
    } else {
        channelSignalChart5 = chart;
    }
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
document.addEventListener('DOMContentLoaded', init); 