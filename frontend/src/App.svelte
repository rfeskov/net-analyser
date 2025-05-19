<script>
  import { Router, Link, Route } from "svelte-routing";
  import Main from "./Main.svelte";
  import Settings from "./Settings.svelte";

  export let url = "";
</script>

<Router {url}>
  <nav class="bg-gray-800 text-white p-4">
    <div class="container mx-auto flex justify-between items-center">
      <Link to="/" class="text-xl font-bold">Wi-Fi Channel Optimizer</Link>
      <div class="space-x-4">
        <Link to="/" class="hover:text-gray-300">Dashboard</Link>
        <Link to="/settings" class="hover:text-gray-300">Settings</Link>
      </div>
    </div>
  </nav>

  <main class="container mx-auto p-4">
    <Route path="/" component={Main} />
    <Route path="/settings" component={Settings} />
  </main>
</Router>

<style>
  :global(body) {
    margin: 0;
    font-family: 'Roboto', sans-serif;
    background-color: #f5f5f5;
  }
</style> 