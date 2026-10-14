# The hosted site: the web app, the league, and the MCP server for chat assistants at /mcp.
# Any container host runs it. Keep a volume at /data: the league and the forecast ledger live there.
FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE llms.txt ./
COPY src ./src
COPY proofs ./proofs
RUN pip install --no-cache-dir -e ".[host]"
ENV EDGEORLUCK_DATA_DIR=/data PYTHONUNBUFFERED=1
EXPOSE 8080
# Behind the host's proxy: the visitor's address (for the rate limits) comes from X-Forwarded-For.
CMD ["sh", "-c", "uvicorn --factory edgeorluck.web.remote:make_app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips '*'"]
