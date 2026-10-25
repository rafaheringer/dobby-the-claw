import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";
import { VitePWA } from "vite-plugin-pwa";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig(({ mode }) => {
  // `vite --mode mock` talks to dev/mock-bridge.mjs instead of the real bridge.
  const BRIDGE = mode === "mock" ? "127.0.0.1:18901" : "127.0.0.1:18900";
  return {
  plugins: [
    react(),
    tailwindcss(),
    // Pollen's robot model is Draco-compressed; the decoder must be served by us (the CSP allows no CDN).
    viteStaticCopy({ targets: [{ src: "node_modules/three/examples/jsm/libs/draco/gltf/draco_{decoder.js,decoder.wasm,wasm_wrapper.js}", dest: "draco" }] }),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["icon.svg", "icon-192.png"],
      manifest: {
        name: "Dobby",
        short_name: "Dobby",
        description: "Controle o Dobby: câmera, conversa, logs e o robô em 3D.",
        lang: "pt-BR",
        start_url: "/",
        scope: "/",
        display: "standalone",
        background_color: "#0f111b",
        theme_color: "#0f111b",
        icons: [
          { src: "/icon-192.png", sizes: "192x192", type: "image/png", purpose: "any" },
          { src: "/icon-512.png", sizes: "512x512", type: "image/png", purpose: "any" },
          { src: "/icon-maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
        ],
      },
      workbox: {
        globPatterns: ["**/*.{js,css,html,svg,png,woff2,glb,wasm}"],
        maximumFileSizeToCacheInBytes: 6_000_000,
        navigateFallback: "/index.html",
        navigateFallbackDenylist: [/^\/ws/, /^\/api\//],
      },
    }),
  ],
  server: {
    proxy: {
      "/ws": { target: `ws://${BRIDGE}`, ws: true },
      "/api": `http://${BRIDGE}`,
    },
  },
  test: { environment: "node" },
  };
});
