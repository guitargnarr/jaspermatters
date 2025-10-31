# 🚀 Deploy FastAPI Backend to Render.com

## Quick Deploy (5 minutes)

### Step 1: Push to GitHub
```bash
cd /tmp/jaspermatters
git add -A
git commit -m "feat(backend): Add FastAPI with ML model endpoints"
git push origin main
```

### Step 2: Deploy on Render

1. Go to: https://render.com
2. Click "New +" → "Web Service"
3. Connect GitHub repository: `guitargnarr/jaspermatters`
4. Configure:
   ```
   Name:           jaspermatters-api
   Region:         Oregon (US West)
   Branch:         main
   Runtime:        Python 3
   Build Command:  pip install -r requirements.txt
   Start Command:  uvicorn backend.api:app --host 0.0.0.0 --port $PORT
   Plan:           Free
   ```
5. Click "Create Web Service"

### Step 3: Wait for Build (2-3 minutes)

Monitor build logs. Should show:
```
Installing dependencies from requirements.txt...
✓ TensorFlow installed
✓ FastAPI installed
==> Your service is live!
```

### Step 4: Get API URL

Your API will be at: `https://jaspermatters-api.onrender.com`

Test it:
```bash
curl https://jaspermatters-api.onrender.com/
# Should return: {"service": "JasperMatters ML API", "status": "healthy"}
```

---

## Configure Frontend to Use Real API

### Option A: Environment Variable (Production)

1. In Netlify dashboard:
   - Site settings → Build & deploy → Environment
   - Add: `VITE_API_URL = https://jaspermatters-api.onrender.com`
   - Add: `VITE_USE_REAL_API = true`

2. Redeploy frontend:
   ```bash
   netlify deploy --prod --dir=dist
   ```

### Option B: Local Development

Create `.env`:
```
VITE_API_URL=http://localhost:8000
VITE_USE_REAL_API=true
```

Run backend locally:
```bash
python -m uvicorn backend.api:app --reload
```

Run frontend:
```bash
npm run dev
```

---

## Test API Endpoints

### 1. Health Check
```bash
curl https://jaspermatters-api.onrender.com/api/health
```

### 2. Salary Prediction
```bash
curl -X POST https://jaspermatters-api.onrender.com/api/predict-salary \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Senior ML Engineer",
    "seniority": "Senior",
    "remote": true,
    "yearsExp": 5,
    "skills": ["Python", "TensorFlow", "Docker"]
  }'
```

### 3. Job Search
```bash
curl -X POST https://jaspermatters-api.onrender.com/api/search-jobs \
  -H "Content-Type: application/json" \
  -d '{
    "query": "senior machine learning engineer remote",
    "top_k": 3
  }'
```

### 4. Skill Analysis
```bash
curl -X POST https://jaspermatters-api.onrender.com/api/analyze-skills \
  -H "Content-Type: application/json" \
  -d '{
    "resume": "Python developer with TensorFlow experience",
    "target_role": "Senior ML Engineer"
  }'
```

---

## Troubleshooting

### Build Fails
**Check:**
- Python version (should be 3.11)
- requirements.txt has all dependencies
- No syntax errors in backend/api.py

### Model Load Fails
**Solution:**
```python
# backend/api.py will gracefully degrade
# Returns 503 if model not loaded
# Frontend falls back to mock data
```

### CORS Errors
**Check:**
- jaspermatters.com in allow_origins list
- Netlify environment variables set
- API URL correct in frontend

---

## Free Tier Limits (Render.com)

- ✅ 750 hours/month (always-on for 1 service)
- ✅ Automatic sleep after 15 min inactivity
- ✅ Cold start: ~30 seconds (first request wakes it)
- ⚠️ Will sleep if unused (acceptable for portfolio)

**Tip:** For demos, first request may be slow (cold start). Have backup tab open during interviews!

---

## Monitoring

**Render Dashboard:**
- View logs: https://dashboard.render.com
- Check metrics: CPU, memory, requests
- Monitor errors

**API Status:**
```bash
watch -n 5 curl https://jaspermatters-api.onrender.com/api/health
```

---

## Success Checklist

- [ ] Backend deployed on Render
- [ ] API returns 200 on health check
- [ ] Frontend environment variables set
- [ ] CORS working (no browser errors)
- [ ] All 3 endpoints tested manually
- [ ] Frontend makes real API calls
- [ ] Cold start acceptable (<30s)

---

**Once deployed, jaspermatters.com will use REAL TensorFlow predictions!** 🎉
