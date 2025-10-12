# CLAUDE.md - Job Intelligence Platform Configuration

## Project Overview
AI-powered job market analysis platform demonstrating production ML engineering skills for Matthew Scott's portfolio. Live at jaspermatters.com.

## Architecture
```
jaspermatters-job-intelligence/
├── backend/
│   ├── scrapers/       # Job data collection (Indeed, LinkedIn, RemoteOK)
│   ├── api/           # FastAPI endpoints
│   └── services/      # Business logic
├── ml/
│   ├── models/        # TensorFlow salary predictor, clustering
│   ├── embeddings/    # Vector search with sentence-transformers
│   └── data/          # Processed job data
├── frontend/          # Next.js 14 + Tailwind
└── deployment/        # Docker, CI/CD configs
```

## Core ML Components

### 1. Vector Search Engine (`ml/embeddings/vector_engine.py`)
- Sentence-transformers for semantic embeddings
- Local index with pickle, ready for Pinecone
- Cosine similarity for job matching
- Skill gap analysis

### 2. Salary Predictor (`ml/models/salary_predictor.py`)
- TensorFlow/Keras neural network
- 134 engineered features
- Handles unseen labels gracefully
- Log-transformed targets for better distribution

### 3. Job Clusterer (`ml/models/job_clusterer.py`)
- K-means for market segmentation
- DBSCAN for outlier detection
- PCA visualization
- Automated optimal cluster selection

### 4. Job Scraper (`backend/scrapers/job_scraper.py`)
- Async scraping from multiple sources
- Structured data extraction (salary, requirements, seniority)
- Deduplication logic
- Rate limiting

## Key Technologies
- **ML/AI**: TensorFlow, scikit-learn, sentence-transformers
- **Backend**: FastAPI, PostgreSQL, Redis
- **Frontend**: Next.js 14, Three.js, D3.js
- **Infrastructure**: Docker, Vercel, Railway

## Development Commands
```bash
# Activate environment
source venv/bin/activate

# Run job scraper
python backend/scrapers/job_scraper.py

# Test vector search
python ml/embeddings/vector_engine.py

# Train salary model
python ml/models/salary_predictor.py

# Run clustering
python ml/models/job_clusterer.py

# Start API (when ready)
uvicorn backend.api.main:app --reload

# Start frontend (when ready)
cd frontend && npm run dev
```

## Current Status
✅ Job scraping pipeline
✅ Vector embeddings system
✅ TensorFlow salary prediction
✅ Clustering & segmentation
🔄 FastAPI backend
🔄 Next.js frontend
🔄 Production deployment

## Performance Metrics
- Vector search: 0.4+ cosine similarity scores
- Salary model: Trained on synthetic + real data
- Clustering: 3 market segments identified
- Scraper: 15+ jobs collected per run

## Next Steps
1. Expand scraping to 1000+ jobs
2. Create FastAPI endpoints
3. Build interactive frontend
4. Deploy to jaspermatters.com
5. Add resume optimization features

## Important Notes
- Models use local storage during development
- Handles unseen labels with fallback mappings
- All credentials in .env (not committed)
- ML models saved as .h5 and .pkl files

## Portfolio Impact
This project demonstrates:
- End-to-end ML pipeline development
- Multiple ML paradigms (supervised, unsupervised, embeddings)
- Production-ready code architecture
- Real-world problem solving
- Modern tech stack proficiency

Target salary justification: $180-220K based on demonstrated ML engineering skills.

---
*Built by Matthew Scott | August 2025 | Part of AI/ML portfolio*