import { writable } from 'svelte/store';

// Access points store
export const accessPoints = writable([
  { id: 1, name: 'AP-1', type: 'real', channel: 1, band: '2.4 GHz' },
  { id: 2, name: 'AP-2', type: 'real', channel: 6, band: '2.4 GHz' },
  { id: 3, name: 'AP-3', type: 'virtual', channel: 11, band: '2.4 GHz' }
]);

// Selected access point for chart
export const selectedAccessPoint = writable(null);

// Channel utilization data
export const channelData = writable({
  labels: [],
  datasets: []
});

// User settings
export const settings = writable({
  updateFrequency: 5, // minutes
  virtualPointsEnabled: true,
  cacheEnabled: true
});

// Chart configuration
export const chartConfig = writable({
  type: 'line',
  options: {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: 'Channel Utilization Over Time'
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Utilization (%)'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Time'
        }
      }
    }
  }
}); 