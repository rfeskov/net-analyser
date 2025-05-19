import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  resolve: {
    alias: {
      'svelte-routing': 'svelte-routing/src/Router.svelte'
    }
  }
}); 