<script>
  import { onMount } from 'svelte';
  import { Line } from 'svelte-chartjs';
  import { channelData, chartConfig, selectedAccessPoint } from '../store.js';

  let chartElement;
  let chart;

  onMount(() => {
    // Initialize with sample data
    channelData.set({
      labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
      datasets: [{
        label: 'Channel Utilization',
        data: [30, 45, 60, 75, 65, 50],
        borderColor: '#4CAF50',
        tension: 0.4
      }]
    });
  });

  $: if ($selectedAccessPoint) {
    // Update chart when selected access point changes
    // This would typically fetch new data from the backend
    console.log('Selected AP changed:', $selectedAccessPoint);
  }
</script>

<div class="bg-white p-4 rounded-lg shadow-lg">
  <div class="mb-4">
    <h2 class="text-xl font-semibold">Channel Utilization</h2>
  </div>
  
  <div class="h-96">
    <Line data={$channelData} options={$chartConfig.options} />
  </div>
</div>

<style>
  :global(.chart-container) {
    position: relative;
    height: 100%;
    width: 100%;
  }
</style> 