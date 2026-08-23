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

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY gunicorn.conf.py ./
COPY prompts/ ./prompts/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

EXPOSE 5001

CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
