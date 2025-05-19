<script>
  import { accessPoints } from '../store.js';

  let newPoint = {
    name: '',
    channel: 1,
    band: '2.4 GHz',
    signalStrength: -65,
    clientCount: 10
  };

  const bands = ['2.4 GHz', '5 GHz'];
  const channels = {
    '2.4 GHz': Array.from({ length: 13 }, (_, i) => i + 1),
    '5 GHz': [36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112,
              116, 120, 124, 128, 132, 136, 140, 144, 149, 153,
              157, 161, 165]
  };

  function addVirtualPoint() {
    if (!newPoint.name) {
      alert('Please enter a name for the virtual access point');
      return;
    }

    const newId = Math.max(...$accessPoints.map(ap => ap.id), 0) + 1;
    const point = {
      id: newId,
      ...newPoint,
      type: 'virtual'
    };

    accessPoints.update(aps => [...aps, point]);
    
    // Reset form
    newPoint = {
      name: '',
      channel: 1,
      band: '2.4 GHz',
      signalStrength: -65,
      clientCount: 10
    };
  }

  function removeVirtualPoint(id) {
    if (confirm('Are you sure you want to remove this virtual access point?')) {
      accessPoints.update(aps => aps.filter(ap => ap.id !== id));
    }
  }
</script>

<div class="bg-white p-6 rounded-lg shadow-lg">
  <h2 class="text-xl font-bold mb-6">Virtual Access Points</h2>

  <!-- Add New Virtual Point Form -->
  <div class="mb-8 p-4 border rounded-lg">
    <h3 class="text-lg font-semibold mb-4">Add New Virtual Point</h3>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-2">Name</label>
        <input
          type="text"
          bind:value={newPoint.name}
          class="w-full p-2 border rounded"
          placeholder="Enter point name"
        />
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 mb-2">Band</label>
        <select
          bind:value={newPoint.band}
          class="w-full p-2 border rounded"
        >
          {#each bands as band}
            <option value={band}>{band}</option>
          {/each}
        </select>
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 mb-2">Channel</label>
        <select
          bind:value={newPoint.channel}
          class="w-full p-2 border rounded"
        >
          {#each channels[newPoint.band] as channel}
            <option value={channel}>{channel}</option>
          {/each}
        </select>
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 mb-2">Signal Strength (dBm)</label>
        <input
          type="number"
          bind:value={newPoint.signalStrength}
          class="w-full p-2 border rounded"
          min="-100"
          max="-30"
        />
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 mb-2">Client Count</label>
        <input
          type="number"
          bind:value={newPoint.clientCount}
          class="w-full p-2 border rounded"
          min="0"
          max="100"
        />
      </div>

      <div class="md:col-span-2">
        <button
          on:click={addVirtualPoint}
          class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Add Virtual Point
        </button>
      </div>
    </div>
  </div>

  <!-- List of Virtual Points -->
  <div>
    <h3 class="text-lg font-semibold mb-4">Existing Virtual Points</h3>
    
    <div class="space-y-4">
      {#each $accessPoints.filter(ap => ap.type === 'virtual') as point}
        <div class="p-4 border rounded-lg flex justify-between items-center">
          <div>
            <h4 class="font-medium">{point.name}</h4>
            <p class="text-sm text-gray-600">
              {point.band} - Channel {point.channel} - {point.signalStrength} dBm - {point.clientCount} clients
            </p>
          </div>
          <button
            on:click={() => removeVirtualPoint(point.id)}
            class="text-red-500 hover:text-red-700"
          >
            Remove
          </button>
        </div>
      {/each}
    </div>
  </div>
</div>

<style>
  input[type="number"] {
    -moz-appearance: textfield;
  }
  input[type="number"]::-webkit-outer-spin-button,
  input[type="number"]::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
</style> 