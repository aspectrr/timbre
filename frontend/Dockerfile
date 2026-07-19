# Static frontend for Fly. Build with bun (matches local dev), serve the
# Vite dist/ with Caddy. Fly terminates TLS at the edge, so Caddy serves
# plain HTTP on the internal port. SPA fallback: unknown paths → index.html.

# ── build ──────────────────────────────────────────────────────────────────
FROM oven/bun:1 AS build
WORKDIR /app
# Install from package.json on the stable bun tag. The repo's bun.lock was
# written by a newer bun (lockfileVersion 2) that the published stable image
# can't parse; for a 3-dep frontend the lockfile adds no reproducibility the
# semver pins don't already give, so install without it.
COPY package.json ./
RUN bun install
COPY . .
# Vite bakes VITE_API_URL at build time — the Fly backend URL. Overridable
# via --build-arg if the backend origin ever moves.
ARG VITE_API_URL=https://style-clone-backend.fly.dev
ENV VITE_API_URL=$VITE_API_URL
RUN bun run build

# ── serve ──────────────────────────────────────────────────────────────────
FROM caddy:2-alpine
COPY Caddyfile /etc/caddy/Caddyfile
COPY --from=build /app/dist /srv
