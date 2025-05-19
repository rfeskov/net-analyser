<script>
  import ChartView from './components/ChartView.svelte';
  import TableView from './components/TableView.svelte';
  import { accessPoints, selectedAccessPoint } from './store.js';

  let selectedBand = '2.4 GHz';
  const bands = ['2.4 GHz', '5 GHz'];
</script>

<div class="grid grid-cols-1 lg:grid-cols-4 gap-4">
  <!-- Sidebar -->
  <div class="lg:col-span-1">
    <div class="bg-white p-4 rounded-lg shadow-lg">
      <h2 class="text-xl font-semibold mb-4">Filters</h2>
      
      <div class="mb-4">
        <label class="block text-sm font-medium text-gray-700 mb-2">Access Point</label>
        <select
          class="w-full p-2 border rounded"
          bind:value={$selectedAccessPoint}
          on:change={() => console.log('Selected AP:', $selectedAccessPoint)}
        >
          <option value={null}>All Access Points</option>
          {#each $accessPoints as ap}
            <option value={ap}>{ap.name}</option>
          {/each}
        </select>
      </div>

      <div class="mb-4">
        <label class="block text-sm font-medium text-gray-700 mb-2">Frequency Band</label>
        <select
          class="w-full p-2 border rounded"
          bind:value={selectedBand}
        >
          {#each bands as band}
            <option value={band}>{band}</option>
          {/each}
        </select>
      </div>
    </div>
  </div>

  <!-- Main Content -->
  <div class="lg:col-span-3 space-y-4">
    <ChartView />
    <TableView />
  </div>
</div>

<style>
  select {
    appearance: none;
    background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e");
    background-repeat: no-repeat;
    background-position: right 0.5rem center;
    background-size: 1em;
  }
</style> 