"""
Test suite for FastAPI backend
Updated to match production API (memory-optimized version)
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.api import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health and status endpoints"""

    def test_root_endpoint(self):
        """Test root health check"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "JasperMatters ML API"
        assert data["status"] == "healthy"
        assert "endpoints" in data

    def test_health_endpoint(self):
        """Test detailed health check"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models" in data

    def test_stats_endpoint(self):
        """Test stats endpoint"""
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "model_features" in data
        assert data["model_features"] == 134


class TestSalaryPrediction:
    """Test salary prediction endpoint"""

    def test_predict_salary_success(self):
        """Test successful salary prediction (accepts fallback or real model)"""
        payload = {
            "title": "Senior ML Engineer",
            "seniority": "Senior",
            "remote": True,
            "yearsExp": 5,
            "skills": ["Python", "TensorFlow", "Docker"]
        }

        response = client.post("/api/predict-salary", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "predicted_salary" in data
        assert "confidence_range" in data
        assert len(data["confidence_range"]) == 2
        assert data["confidence_range"][0] < data["predicted_salary"]
        assert data["confidence_range"][1] > data["predicted_salary"]

        # Verify info field exists
        assert "info" in data
        assert "using_real_model" in data["info"]

    def test_predict_salary_validation(self):
        """Test input validation"""
        # Missing required field
        payload = {
            "title": "Engineer",
            "seniority": "Senior"
            # missing remote, yearsExp, skills
        }

        response = client.post("/api/predict-salary", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_salary_invalid_seniority(self):
        """Test invalid seniority level"""
        payload = {
            "title": "Engineer",
            "seniority": "SuperSenior",  # Invalid
            "remote": True,
            "yearsExp": 5,
            "skills": ["Python"]
        }

        response = client.post("/api/predict-salary", json=payload)
        assert response.status_code == 422

    def test_predict_salary_range_check(self):
        """Test predicted salary is in reasonable range"""
        payload = {
            "title": "Junior Engineer",
            "seniority": "Junior",
            "remote": True,
            "yearsExp": 1,
            "skills": ["Python"]
        }

        response = client.post("/api/predict-salary", json=payload)
        assert response.status_code == 200

        data = response.json()
        # Junior salary should be under $150K
        assert data["predicted_salary"] < 150000
        # But above $40K (minimum reasonable)
        assert data["predicted_salary"] > 40000


class TestSkillAnalysis:
    """Test skill gap analysis endpoint"""

    def test_analyze_skills_success(self):
        """Test successful skill analysis"""
        payload = {
            "resume": "Python developer with TensorFlow and Docker experience",
            "target_role": "Senior ML Engineer"
        }

        response = client.post("/api/analyze-skills", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "matching_skills" in data
        assert "missing_skills" in data
        assert "match_percentage" in data
        assert 0 <= data["match_percentage"] <= 100

    def test_analyze_skills_validation(self):
        """Test resume length validation"""
        payload = {
            "resume": "short",  # Too short (min 10 chars)
            "target_role": "Senior ML Engineer"
        }

        response = client.post("/api/analyze-skills", json=payload)
        assert response.status_code == 422

    def test_analyze_skills_matching(self):
        """Test skill matching logic"""
        payload = {
            "resume": "Expert in Python, TensorFlow, PyTorch, Docker, Kubernetes, AWS, and SQL",
            "target_role": "Senior ML Engineer"
        }

        response = client.post("/api/analyze-skills", json=payload)
        assert response.status_code == 200

        data = response.json()
        # Should match all required skills
        assert data["match_percentage"] == 100.0
        assert len(data["missing_skills"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
