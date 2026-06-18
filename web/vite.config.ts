import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base must match the GitHub Pages project path: user.github.io/forest-leafwood-seg/
// override at build time with:  BASE_PATH=/your-repo/ npm run build
export default defineConfig({
  plugins: [react()],
  base: process.env.BASE_PATH || "/forest-leafwood-seg/",
});
