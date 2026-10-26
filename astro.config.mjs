import { defineConfig } from "astro/config";
import react from "@astrojs/react";

// BASE_PATH permite publicar num subcaminho (por exemplo GitHub Pages: /castelos-site).
const base = process.env.BASE_PATH || "/";

export default defineConfig({
  site: process.env.SITE_URL || "https://castelos.site",
  base,
  trailingSlash: "always",
  integrations: [react()],
  vite: {
    ssr: { noExternal: ["three"] },
  },
});
