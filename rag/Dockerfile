# Stage 1: Build React frontend
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --legacy-peer-deps
COPY frontend/ ./
# Vite inlines VITE_* at build time, so these must arrive as build args --
# setting them as runtime env vars on Railway has no effect on the bundle.
# On Railway: Settings > Build > add these as build arguments.
ARG VITE_SUPABASE_URL
ARG VITE_SUPABASE_ANON_KEY
ENV VITE_SUPABASE_URL=$VITE_SUPABASE_URL
ENV VITE_SUPABASE_ANON_KEY=$VITE_SUPABASE_ANON_KEY
RUN npm run build

# Stage 2: Python backend + built frontend
FROM python:3.11-slim
WORKDIR /app

# Without this, print() goes to block-buffered stdout and app-level logs
# never reach the platform — gunicorn's own stderr lines appear while the
# application's are swallowed, which makes a boot failure look like silence.
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY gunicorn.conf.py ./
COPY prompts/ ./prompts/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Must match the port gunicorn actually binds. Railway injects PORT=8080 and
# routes to the EXPOSEd port when the service has no explicit target port
# set, so declaring 5001 here sent traffic to a port nothing listened on --
# an instant connection-refused 502 against a container reporting healthy.
EXPOSE 8080

CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
