# Q-FinOpt Python Backend API

FastAPI backend serving real-time mutual fund data, Monte Carlo withdrawal simulations, SIP calculators, and PDF report generation.

---

## 🚀 Running Locally

### 1. Set Up Virtual Environment
```bash
cd backend
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Server
```bash
python run_server.py
```
The API will be live at:
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/health`

---

## 🌐 Deploying to the Cloud

### Option A: Render (Recommended, Free / Low Cost)
1. Push your repo to GitHub.
2. Go to [Render Dashboard](https://dashboard.render.com/) -> **New Web Service**.
3. Select your repository and set:
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Copy your live Render URL (e.g. `https://qfinopt-api.onrender.com`) and paste it into `android/app/src/main/java/com/qfinopt/app/config/AppConfig.kt`.

### Option B: Railway
1. Go to [Railway.app](https://railway.app/).
2. Create **New Project from GitHub Repo**.
3. Set the root directory to `backend`.
4. Railway will automatically detect Python and start the FastAPI service.

### Option C: Docker
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
Build & run:
```bash
docker build -t qfinopt-backend .
docker run -p 8000:8000 qfinopt-backend
```
