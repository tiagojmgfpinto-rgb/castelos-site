import { defineConfig } from "@playwright/test";

// Testes no browser sobre o site compilado (dist). No CI o astro preview serve-o com o BASE_PATH do GitHub Pages;
// localmente pode apontar-se para outro servidor com E2E_URL (p. ex. um que vá buscar /terrain/ ao site publicado).
const base = process.env.BASE_PATH || "/";
const url = process.env.E2E_URL || `http://127.0.0.1:4321${base}`;

export default defineConfig({
  testDir: "e2e",
  testMatch: "*.e2e.ts",
  timeout: 180_000,
  expect: { timeout: 30_000 },
  workers: 1,
  retries: 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: url,
    trace: "off",
    launchOptions: { args: ["--use-gl=angle", "--use-angle=swiftshader", "--ignore-gpu-blocklist"] },
  },
  webServer: process.env.E2E_URL
    ? undefined
    : { command: "npx astro preview --host 127.0.0.1 --port 4321", url, reuseExistingServer: false, timeout: 60_000 },
});
