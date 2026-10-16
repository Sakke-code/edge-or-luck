# The hosted site: the web app and the MCP server for chat assistants at /mcp.
# Any container host runs it. Keep a volume at /data: the forecast ledger and the usage counts live there.
FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE llms.txt ./
COPY src ./src
COPY proofs ./proofs
RUN pip install --no-cache-dir -e ".[host]"
ENV EDGEORLUCK_DATA_DIR=/data EDGEORLUCK_PROXIES=1 PYTHONUNBUFFERED=1
EXPOSE 8080
# Behind the host's one proxy (EDGEORLUCK_PROXIES=1): the visitor's address, for the rate limits, is the one that
# proxy added to X-Forwarded-For, never the part a visitor can write. So uvicorn itself trusts no proxy header.
# No access log: a link's address can carry a visitor's trades (/api/judge?pnl=...).
CMD ["sh", "-c", "uvicorn --factory edgeorluck.web.remote:make_app --host 0.0.0.0 --port ${PORT:-8080} --no-proxy-headers --no-access-log"]
