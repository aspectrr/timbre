import { defineConfig } from "vite";
import solid from "vite-plugin-solid";

export default defineConfig({
  plugins: [solid()],
  server: {
    // Proxy API to the local Fly backend during dev.
    proxy: {
      "/api": {
        target: process.env.VITE_API_URL || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
