// Load available points
async function loadPoints() {
    try {
        const response = await fetch('/api/points');
        const points = await response.json();
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
                <a 
                    href="/points/${point.id}"
                    class="block w-full bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 transition-colors duration-200 text-center"
                >
                    Подробнее
                </a>
            </div>
        `;
        
        grid.appendChild(card);
    });
}

// Initialize the page
document.addEventListener('DOMContentLoaded', loadPoints); 