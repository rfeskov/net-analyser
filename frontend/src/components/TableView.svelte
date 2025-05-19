<script>
  import { accessPoints } from '../store.js';

  // Sample data for the table
  let scheduleData = [
    {
      startTime: '08:00',
      endTime: '12:00',
      channel: 1,
      signalStrength: -65,
      clientCount: 15
    },
    {
      startTime: '12:00',
      endTime: '16:00',
      channel: 6,
      signalStrength: -70,
      clientCount: 20
    },
    {
      startTime: '16:00',
      endTime: '20:00',
      channel: 11,
      signalStrength: -68,
      clientCount: 18
    }
  ];

  let sortColumn = 'startTime';
  let sortDirection = 'asc';

  function sortTable(column) {
    if (sortColumn === column) {
      sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
      sortColumn = column;
      sortDirection = 'asc';
    }

    scheduleData = [...scheduleData].sort((a, b) => {
      const aVal = a[column];
      const bVal = b[column];
      
      if (sortDirection === 'asc') {
        return aVal > bVal ? 1 : -1;
      } else {
        return aVal < bVal ? 1 : -1;
      }
    });
  }
</script>

<div class="bg-white p-4 rounded-lg shadow-lg">
  <div class="mb-4">
    <h2 class="text-xl font-semibold">Channel Switching Schedule</h2>
  </div>

  <div class="overflow-x-auto">
    <table class="min-w-full">
      <thead>
        <tr class="bg-gray-100">
          <th class="px-4 py-2 cursor-pointer" on:click={() => sortTable('startTime')}>
            Start Time {sortColumn === 'startTime' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
          </th>
          <th class="px-4 py-2 cursor-pointer" on:click={() => sortTable('endTime')}>
            End Time {sortColumn === 'endTime' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
          </th>
          <th class="px-4 py-2 cursor-pointer" on:click={() => sortTable('channel')}>
            Channel {sortColumn === 'channel' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
          </th>
          <th class="px-4 py-2 cursor-pointer" on:click={() => sortTable('signalStrength')}>
            Signal Strength {sortColumn === 'signalStrength' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
          </th>
          <th class="px-4 py-2 cursor-pointer" on:click={() => sortTable('clientCount')}>
            Client Count {sortColumn === 'clientCount' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
          </th>
        </tr>
      </thead>
      <tbody>
        {#each scheduleData as row}
          <tr class="border-b hover:bg-gray-50">
            <td class="px-4 py-2">{row.startTime}</td>
            <td class="px-4 py-2">{row.endTime}</td>
            <td class="px-4 py-2">{row.channel}</td>
            <td class="px-4 py-2">{row.signalStrength} dBm</td>
            <td class="px-4 py-2">{row.clientCount}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</div>

<style>
  th {
    text-align: left;
  }
</style> 