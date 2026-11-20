import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The API runs on loopback (default 127.0.0.1:8010, set PROPHET_API_URL to override). In dev, Vite proxies /api to it
// so the page and API share an origin. The API accepts writes from any loopback origin, so Vite may pick another port.
const apiTarget = process.env.PROPHET_API_URL ?? "http://127.0.0.1:8010";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": { target: apiTarget, changeOrigin: false } },
  },
  build: { outDir: "dist", emptyOutDir: true, sourcemap: false },
});
