// Load available points
async function loadPoints() {
    try {
        const response = await fetch('/api/points');
        const points = await response.json();
        console.log('Loaded points from API:', points);
        
        // Add static points only on the points page
        const staticPoints = [
            {
                id: "point_1",
                name: "point_1",
                band: "2.4 GHz & 5 GHz",
                is_online: false,
                clients_count: 0,
                channel: "Авто",
                channel_24: "Авто",
                channel_5: "Авто",
                power: "100",
                is_static: true
            },
            {
                id: "point_2",
                name: "point_2",
                band: "2.4 GHz & 5 GHz",
                is_online: false,
                clients_count: 0,
                channel: "Авто",
                channel_24: "Авто",
                channel_5: "Авто",
                power: "100",
                is_static: true
            }
        ];
        console.log('Static points:', staticPoints);

        // Combine static points with dynamic points
        const allPoints = [...staticPoints, ...points];
        console.log('All points:', allPoints);
        updatePointsGrid(allPoints);
    } catch (error) {
        console.error('Ошибка загрузки точек:', error);
    }
}

// Update points grid with cards
function updatePointsGrid(points) {
    const grid = document.getElementById('points-grid');
    if (!grid) {
        console.error('Grid element not found');
        return;
    }

    console.log('Updating grid with points:', points);
    grid.innerHTML = '';

    points.forEach(point => {
        const card = createPointCard(point);
        grid.appendChild(card);
    });
}

function createPointCard(point) {
    console.log('Creating card for point:', point);
    const card = document.createElement('div');
    card.className = 'point-card';
    card.dataset.pointId = point.id;

    const statusClass = point.is_online ? 'status-online' : 'status-offline';
    const statusText = point.is_online ? 'Онлайн' : 'Оффлайн';

    // Create base card content without settings button
    let cardContent = `
        <div class="point-header">
            <h3>${point.name}</h3>
            <span class="status ${statusClass}">${statusText}</span>
        </div>
        <div class="point-info">
            <p><strong>Диапазон:</strong> ${point.band}</p>
            <p><strong>Канал:</strong> ${point.channel}</p>
            <p><strong>Мощность сигнала:</strong> ${point.signal_strength !== null ? point.signal_strength + ' dBm' : 'Н/Д'}</p>
        </div>
    `;

    // Add settings button only for non-static points
    if (!point.is_static) {
        cardContent += `
            <div class="point-actions">
                <button class="settings-btn">
                    <i class="fas fa-cog"></i> Настройки
                </button>
            </div>
        `;
    }

    card.innerHTML = cardContent;

    // Add click handler for settings button only if not static
    if (!point.is_static) {
        const settingsBtn = card.querySelector('.settings-btn');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', () => openSettingsModal(point.id));
        }
    }

    return card;
}

// Modal functionality
let currentPointId = null;

function openSettingsModal(pointId) {
    currentPointId = pointId;
    const modal = document.getElementById('settings-modal');
    const title = document.getElementById('modal-title');
    title.textContent = `Настройки точки доступа ${pointId}`;
    
    // Load point settings
    loadPointSettings(pointId);
    
    // Show modal
    modal.classList.remove('hidden');
    
    // Add event listeners for channel mode changes
    document.getElementById('channel-mode-24').addEventListener('change', function(e) {
        document.getElementById('channel-24').disabled = e.target.value === 'auto';
    });
    
    document.getElementById('channel-mode-5').addEventListener('change', function(e) {
        document.getElementById('channel-5').disabled = e.target.value === 'auto';
    });
}

function closeSettingsModal() {
    const modal = document.getElementById('settings-modal');
    modal.classList.add('hidden');
    currentPointId = null;
}

// Add function to save settings to JSON file
async function saveSettingsToFile(pointId, settings) {
    try {
        const response = await fetch('/api/settings/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                pointId: pointId,
                settings: settings
            })
        });

        if (!response.ok) {
            throw new Error('Failed to save settings');
        }
    } catch (error) {
        console.error('Error saving settings to file:', error);
        throw error;
    }
}

// Add function to load settings from JSON file
async function loadSettingsFromFile(pointId) {
    try {
        const response = await fetch('/api/settings/load');
        if (!response.ok) {
            throw new Error('Failed to load settings');
        }
        const data = await response.json();
        return data.points[pointId] || null;
    } catch (error) {
        console.error('Error loading settings from file:', error);
        return null;
    }
}

// Update loadPointSettings to use JSON file
async function loadPointSettings(pointId) {
    try {
        // Get current points data
        const response = await fetch('/api/points');
        const points = await response.json();
        const point = points.find(p => p.id === pointId);
        
        if (!point) {
            throw new Error('Point not found');
        }

        // Try to load saved settings first
        const savedSettings = await loadSettingsFromFile(pointId);
        
        // Set default values based on current point data or saved settings
        document.getElementById('point-status').checked = savedSettings?.status ?? point.is_online;
        
        // Set 2.4 GHz settings
        const channel24 = savedSettings?.band24?.channel || point.channel_24 || point.channel;
        const isAuto24 = savedSettings?.band24?.channelMode === 'auto' || !channel24 || channel24 === 'Авто';
        document.getElementById('channel-mode-24').value = isAuto24 ? 'auto' : 'manual';
        document.getElementById('channel-24').value = isAuto24 ? '1' : channel24;
        document.getElementById('channel-24').disabled = isAuto24;
        document.getElementById('power-24').value = savedSettings?.band24?.power || '100';
        
        // Set 5 GHz settings
        const channel5 = savedSettings?.band5?.channel || point.channel_5;
        const isAuto5 = savedSettings?.band5?.channelMode === 'auto' || !channel5 || channel5 === 'Авто';
        document.getElementById('channel-mode-5').value = isAuto5 ? 'auto' : 'manual';
        document.getElementById('channel-5').value = isAuto5 ? '36' : channel5;
        document.getElementById('channel-5').disabled = isAuto5;
        document.getElementById('power-5').value = savedSettings?.band5?.power || '100';

    } catch (error) {
        console.error('Error loading point settings:', error);
        alert('Ошибка при загрузке настроек точки');
    }
}

// Update saveSettings to use JSON file
async function saveSettings() {
    if (!currentPointId) return;
    
    const settings = {
        pointId: currentPointId,
        status: document.getElementById('point-status').checked,
        band24: {
            channelMode: document.getElementById('channel-mode-24').value,
            channel: document.getElementById('channel-24').value,
            power: document.getElementById('power-24').value
        },
        band5: {
            channelMode: document.getElementById('channel-mode-5').value,
            channel: document.getElementById('channel-5').value,
            power: document.getElementById('power-5').value
        }
    };
    
    try {
        // Save settings to JSON file
        await saveSettingsToFile(currentPointId, settings);
        
        // Get current points data
        const pointsResponse = await fetch('/api/points');
        const points = await pointsResponse.json();
        
        // Update the point in the grid while preserving all other data
        const updatedPoints = points.map(point => {
            if (point.id === currentPointId) {
                return {
                    ...point,
                    is_online: settings.status,
                    channel_24: settings.band24.channelMode === 'auto' ? 'Авто' : settings.band24.channel,
                    channel_5: settings.band5.channelMode === 'auto' ? 'Авто' : settings.band5.channel,
                    channel: point.channel || settings.band24.channelMode === 'auto' ? 'Авто' : settings.band24.channel,
                    power: settings.band24.power // Use 2.4 GHz power as the main power level
                };
            }
            return point;
        });
        
        updatePointsGrid(updatedPoints);
        closeSettingsModal();
    } catch (error) {
        console.error('Error saving settings:', error);
        alert('Ошибка при сохранении настроек');
    }
}

// Initialize the page
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, calling loadPoints');
    loadPoints();
}); 