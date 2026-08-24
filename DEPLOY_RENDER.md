# Deploying Smart Scan Strategy to Render

This repository is fully configured for **1-click / zero-config deployment on Render** using Render's free tier.

---

## Architecture Overview

- **Unified Full-Stack Deployment**: The Python web service builds the modern React / Vite frontend (`frontend/dist`) and serves both the REST API (`/api/*`) and the Single Page Application UI from a single service.
- **Port & Host**: Uses dynamic `$PORT` and binds to `0.0.0.0`.
- **Infrastructure as Code**: Preconfigured via `render.yaml`.

---

## Method 1: Automatic Deployment using Render Blueprint (Recommended)

1. **Push your code to GitHub / GitLab**:
   ```bash
   git add .
   git commit -m "Configure project for Render deployment"
   git push origin main
   ```

2. **Open Render Dashboard**:
   - Go to [dashboard.render.com](https://dashboard.render.com/)
   - Click **"New +"** (top right) and select **"Blueprint"**

3. **Connect Your Repository**:
   - Select your `SIH2` repository
   - Render will automatically detect `render.yaml` and configure:
     - **Service Name**: `ew-smart-scan`
     - **Runtime**: `Python 3.11.9`
     - **Build Command**: `npm --prefix frontend install && npm --prefix frontend run build && pip install -r requirements.txt`
     - **Start Command**: `python -m app.server`

4. **Click "Apply"**:
   - Render will build the React assets, install Python packages, and deploy the live web service.
   - Once deployed, your site will be live at `https://<your-service-name>.onrender.com`.

---

## Method 2: Manual Web Service Creation on Render

If you prefer to configure the Web Service manually in the Render UI:

1. Go to [dashboard.render.com](https://dashboard.render.com/)
2. Click **"New +"** -> **"Web Service"**
3. Connect your GitHub repository
4. Fill in the settings:
   - **Name**: `ew-smart-scan` (or your preferred name)
   - **Region**: Choose the closest region (e.g., `Oregon (US West)` or `Frankfurt (EU)`)
   - **Branch**: `main`
   - **Runtime**: `Python`
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     python -m app.server
     ```
   - **Instance Type**: `Free`
5. **Environment Variables** (under *Advanced*):
   - `PYTHON_VERSION`: `3.11.9`
   - `HOST`: `0.0.0.0`
6. Click **"Create Web Service"**.

---

## Verification & Health Check

Once deployed, test your live URL:
- **Command Center & UI**: `https://<your-app>.onrender.com/`
- **Dashboard**: `https://<your-app>.onrender.com/dashboard`
- **API Status**: `https://<your-app>.onrender.com/api/status`
- **Prototype Dataset API**: `https://<your-app>.onrender.com/api/prototype/dataset`

---

## Troubleshooting

- **Node/NPM in Python runtime**: Render's standard Python environment has Node.js and NPM pre-installed, allowing the build command `npm --prefix frontend run build` to execute seamlessly.
- **Cold Starts on Free Tier**: Render's free tier spins down services after 15 minutes of inactivity. The first request after sleep may take ~30-50 seconds to boot up.
