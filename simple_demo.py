#!/usr/bin/env python3
"""
Job Intelligence Platform - Simple Demo
Shows the core ML capabilities
"""

import json
import pandas as pd
import numpy as np
print("=" * 60)
print("🚀 JOB INTELLIGENCE PLATFORM - ML CAPABILITIES")
print("=" * 60)

# Load job data
print("\n📊 Loading job data...")
with open('ml/data/jobs_data.json', 'r') as f:
    jobs = json.load(f)
print(f"✅ Loaded {len(jobs)} jobs from RemoteOK")

# Show sample jobs
print("\n📋 Sample Jobs in Database:")
for i, job in enumerate(jobs[:3]):
    print(f"\n  {i+1}. {job['title']} at {job['company']}")
    print(f"     Level: {job.get('seniority_level', 'N/A')}")
    if job.get('salary_min'):
        print(f"     Salary: ${job['salary_min']:,.0f} - ${job['salary_max']:,.0f}")
    print(f"     Remote: {'Yes' if job.get('remote') else 'No'}")

# Show ML models available
print("\n" + "=" * 60)
print("🧠 TRAINED ML MODELS AVAILABLE:")
print("=" * 60)

import os
models_dir = 'ml/models/'
models = [f for f in os.listdir(models_dir) if f.endswith(('.h5', '.pkl'))]

for model in models:
    size = os.path.getsize(os.path.join(models_dir, model))
    print(f"\n  ✅ {model}")
    print(f"     Size: {size/1024:.1f} KB")
    
    if 'salary' in model:
        print("     Type: TensorFlow Neural Network (134 features)")
        print("     Purpose: Predict salaries based on job features")
    elif 'cluster' in model:
        print("     Type: K-means & DBSCAN clustering")
        print("     Purpose: Segment job market into categories")
    elif 'preprocessors' in model:
        print("     Type: Feature encoders and scalers")
        print("     Purpose: Transform raw data for ML models")

# Show data statistics
print("\n" + "=" * 60)
print("📊 JOB MARKET STATISTICS:")
print("=" * 60)

jobs_df = pd.DataFrame(jobs)

# Salary statistics
salaries = jobs_df[jobs_df['salary_min'].notna()]
if len(salaries) > 0:
    avg_min = salaries['salary_min'].mean()
    avg_max = salaries['salary_max'].mean()
    print(f"\n💰 Salary Range: ${avg_min:,.0f} - ${avg_max:,.0f}")

# Seniority distribution
seniority_counts = jobs_df['seniority_level'].value_counts()
print("\n👔 Seniority Distribution:")
for level, count in seniority_counts.items():
    print(f"   {level}: {count} jobs ({count/len(jobs_df)*100:.1f}%)")

# Remote work
remote_pct = jobs_df['remote'].mean() * 100
print(f"\n🏠 Remote Jobs: {remote_pct:.1f}%")

# Top skills
all_requirements = []
for reqs in jobs_df['requirements']:
    if isinstance(reqs, list):
        all_requirements.extend(reqs)

if all_requirements:
    from collections import Counter
    skill_counts = Counter(all_requirements)
    print("\n🔧 Top Skills in Demand:")
    for skill, count in skill_counts.most_common(5):
        print(f"   {skill}: {count} mentions")

print("\n" + "=" * 60)
print("🎯 PROJECT ACHIEVEMENTS:")
print("=" * 60)

achievements = [
    "✅ Built job scraping pipeline (3 sources)",
    "✅ Trained TensorFlow neural network (134 features)",
    "✅ Implemented vector embeddings for semantic search",
    "✅ Created K-means clustering for market segmentation",
    "✅ Developed skill gap analysis system",
    "✅ Saved trained models for production use"
]

for achievement in achievements:
    print(f"\n  {achievement}")

print("\n" + "=" * 60)
print("💼 PORTFOLIO IMPACT:")
print("=" * 60)
print("\nThis project demonstrates:")
print("• End-to-end ML pipeline development")
print("• Multiple ML paradigms (supervised & unsupervised)")
print("• Production-ready code architecture")
print("• Real-world problem solving")
print("• Modern tech stack (TensorFlow, scikit-learn, etc.)")

print("\n🎯 Target Roles: $180-220K AI/ML Engineer positions")
print("🌐 Deployment Target: jaspermatters.com")

print("\n" + "=" * 60)
print("✅ Repository ready for GitHub!")
print("=" * 60)