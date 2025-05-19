<script>
  import { settings } from './store.js';
  import VirtualPoints from './components/VirtualPoints.svelte';

  let updateFrequency = $settings.updateFrequency;
  let virtualPointsEnabled = $settings.virtualPointsEnabled;
  let cacheEnabled = $settings.cacheEnabled;

  function saveSettings() {
    settings.set({
      updateFrequency,
      virtualPointsEnabled,
      cacheEnabled
    });
  }

  function clearCache() {
    if (confirm('Are you sure you want to clear the cache?')) {
      // Implement cache clearing logic here
      console.log('Cache cleared');
    }
  }
</script>

<div class="max-w-2xl mx-auto">
  <div class="bg-white p-6 rounded-lg shadow-lg">
    <h1 class="text-2xl font-bold mb-6">Settings</h1>

    <div class="space-y-6">
      <!-- Update Frequency -->
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-2">
          Update Frequency (minutes)
        </label>
        <input
          type="number"
          min="1"
          max="60"
          bind:value={updateFrequency}
          class="w-full p-2 border rounded"
        />
      </div>

      <!-- Virtual Points -->
      <div>
        <label class="flex items-center space-x-2">
          <input
            type="checkbox"
            bind:checked={virtualPointsEnabled}
            class="rounded text-blue-600"
          />
          <span class="text-sm font-medium text-gray-700">
            Enable Virtual Access Points
          </span>
        </label>
      </div>

      <!-- Cache -->
      <div>
        <label class="flex items-center space-x-2">
          <input
            type="checkbox"
            bind:checked={cacheEnabled}
            class="rounded text-blue-600"
          />
          <span class="text-sm font-medium text-gray-700">
            Enable Data Caching
          </span>
        </label>
      </div>

      <!-- Cache Clear Button -->
      <div>
        <button
          on:click={clearCache}
          class="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
        >
          Clear Cache
        </button>
      </div>

      <!-- Save Button -->
      <div>
        <button
          on:click={saveSettings}
          class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Save Settings
        </button>
      </div>
    </div>
  </div>

  <!-- Virtual Points Configuration -->
  {#if virtualPointsEnabled}
    <div class="mt-8">
      <VirtualPoints />
    </div>
  {/if}
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