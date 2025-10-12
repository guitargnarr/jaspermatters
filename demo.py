#!/usr/bin/env python3
"""
Job Intelligence Platform - Full Demo
Demonstrates all ML capabilities
"""

import json
import pandas as pd
from ml.embeddings.vector_engine import VectorEngine
from ml.models.salary_predictor import SalaryPredictor
from ml.models.job_clusterer import JobMarketClusterer

print("=" * 60)
print("🚀 JOB INTELLIGENCE PLATFORM - ML DEMO")
print("=" * 60)

# Load job data
print("\n📊 Loading job data...")
with open('ml/data/jobs_data.json', 'r') as f:
    jobs = json.load(f)
print(f"✅ Loaded {len(jobs)} jobs")

# 1. VECTOR SEARCH DEMO
print("\n" + "=" * 60)
print("1️⃣  SEMANTIC JOB SEARCH")
print("=" * 60)

engine = VectorEngine()
engine.index_jobs('ml/data/jobs_data.json')

query = "I want a senior machine learning role with good pay"
results = engine.search_jobs(query, top_k=3)

print(f"\nQuery: '{query}'")
print("\nTop Matches:")
for i, result in enumerate(results, 1):
    print(f"\n  {i}. {result.title} at {result.company}")
    print(f"     Similarity Score: {result.score:.3f}")
    if result.salary_range[0]:
        print(f"     Salary: ${result.salary_range[0]:,.0f} - ${result.salary_range[1]:,.0f}")

# 2. SALARY PREDICTION DEMO
print("\n" + "=" * 60)
print("2️⃣  SALARY PREDICTION MODEL")
print("=" * 60)

predictor = SalaryPredictor()
predictor.load_model()

test_job = {
    'title': 'Senior ML Engineer',
    'company': 'Tech Startup',
    'description': 'Build ML systems at scale',
    'requirements': ['Python', 'TensorFlow', 'Docker', 'AWS'],
    'seniority_level': 'Senior',
    'remote': True,
    'source': 'RemoteOK',
    'job_type': 'Full-time'
}

prediction = predictor.predict(pd.DataFrame([test_job]))[0]
print(f"\nTest Job: {test_job['title']}")
print(f"Requirements: {', '.join(test_job['requirements'])}")
print(f"💰 Predicted Salary: ${prediction:,.0f}")

# 3. MARKET CLUSTERING DEMO
print("\n" + "=" * 60)
print("3️⃣  JOB MARKET SEGMENTATION")
print("=" * 60)

clusterer = JobMarketClusterer()
jobs_df = pd.DataFrame(jobs)
X = clusterer.extract_clustering_features(jobs_df)
labels = clusterer.perform_kmeans_clustering(X, n_clusters=3)
analysis = clusterer.analyze_clusters(jobs_df, labels)

print("\nMarket Segments Identified:")
for cluster_name, stats in analysis.items():
    print(f"\n  {cluster_name}:")
    print(f"    • Size: {stats['size']} jobs")
    print(f"    • Avg Salary: ${stats['avg_salary']:,.0f}")
    print(f"    • Level: {stats['dominant_seniority']}")
    print(f"    • Top Skills: {', '.join(stats['top_skills'][:3]) if stats['top_skills'] else 'N/A'}")

# 4. SKILL GAP ANALYSIS
print("\n" + "=" * 60)
print("4️⃣  SKILL GAP ANALYSIS")
print("=" * 60)

resume = """
Experienced Python developer with 5 years building web applications.
Strong in Django, REST APIs, PostgreSQL, and Docker.
Some experience with machine learning using scikit-learn.
"""

job_ids = [jobs[0].get('job_id', '0'), jobs[1].get('job_id', '1')]
gap_analysis = engine.skill_gap_analysis(resume, job_ids)

print(f"\nYour Skills: {', '.join(gap_analysis['matching_skills'][:5])}")
print(f"Missing Skills: {', '.join(gap_analysis['missing_skills'][:5]) if gap_analysis['missing_skills'] else 'None'}")
print(f"Match Percentage: {gap_analysis['match_percentage']:.1f}%")

if gap_analysis['priority_skills']:
    print("\nTop Skills to Learn:")
    for skill, frequency in gap_analysis['priority_skills'][:3]:
        print(f"  • {skill} (needed by {frequency} jobs)")

print("\n" + "=" * 60)
print("✅ DEMO COMPLETE - All ML models working!")
print("=" * 60)
print("\nThis platform demonstrates:")
print("• Semantic search with embeddings")
print("• Neural network salary prediction")
print("• Unsupervised clustering")
print("• Skill gap analysis")
print("\n🎯 Ready for production deployment to jaspermatters.com!")