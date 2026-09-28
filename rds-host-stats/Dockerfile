FROM oven/bun:1 AS deps
WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile --production

FROM oven/bun:1
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY package.json tsconfig.json ./
COPY bin ./bin
COPY lib ./lib
USER bun
CMD ["bun", "run", "./bin/rds-host-stats.ts"]
