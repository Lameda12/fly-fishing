import { defineConfig } from "vite";

// Relative base so the built site works from any path on a static host.
// Nothing here reaches the network at build or run time: three comes from
// node_modules and the recordings are files in public/.
export default defineConfig({
  base: "./",
  build: { outDir: "dist", assetsDir: "assets", sourcemap: false },
  server: { port: 5173, strictPort: false },
});
