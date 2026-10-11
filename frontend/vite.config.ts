import { defineConfig, loadEnv } from "vite";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "RUNNER_");
  const origin = env.RUNNER_LOCAL_ORIGIN;
  const endpoint = env.RUNNER_LOCAL_ENDPOINT;
  const token = env.RUNNER_DESKTOP_TOKEN;
  const local = Boolean(origin && endpoint && token);
  return {
    base: "./",
    build: { outDir: "../src/runner_desktop/web", emptyOutDir: true },
    server: { port: 5173, strictPort: true, cors: false },
    plugins: [
      tailwindcss(),
      {
        name: "local-runner-bootstrap",
        apply: "serve",
        configureServer(server) {
          if (!local) return;
          if (
            new URL(origin).hostname !== "127.0.0.1" ||
            new URL(endpoint).hostname !== "127.0.0.1"
          )
            throw new Error(
              "Automatic authentication requires loopback endpoints",
            );
          server.middlewares.use((request, response, next) => {
            if (
              request.headers.host !== new URL(origin).host ||
              (request.headers.origin && request.headers.origin !== origin) ||
              request.headers["sec-fetch-site"] === "cross-site"
            ) {
              response.statusCode = 403;
              response.end("Forbidden");
              return;
            }
            response.setHeader("Cache-Control", "no-store");
            response.setHeader("Referrer-Policy", "no-referrer");
            next();
          });
        },
        async transformIndexHtml(html) {
          if (!local) return html;
          const response = await fetch(env.RUNNER_TEMPLATE_URL, {
            method: "POST",
            headers: {
              Authorization: `Bearer ${token}`,
              "Content-Type": "text/html",
            },
            body: html,
            signal: AbortSignal.timeout(5000),
          });
          if (!response.ok) throw new Error("Local HTML rendering failed");
          return response.text();
        },
      },
    ],
  };
});
