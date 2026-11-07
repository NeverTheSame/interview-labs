FROM ghcr.io/astral-sh/uv:0.8.22 AS uv
FROM python:3.11-slim
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock README.md ./
COPY r3 ./r3
RUN uv sync --frozen --no-dev && useradd --uid 10001 --create-home portal
ENV PATH="/app/.venv/bin:$PATH"
USER portal
EXPOSE 8090
CMD ["r3-portal"]
