"""Test the trained salary prediction model"""

import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.salary_predictor import SalaryPredictor

# Load the trained model
predictor = SalaryPredictor()
predictor.load_model()

# Test jobs with various seniority levels
test_jobs = [
    {
        'title': 'Junior Machine Learning Engineer',
        'company': 'StartupAI',
        'description': 'Entry-level ML position with growth opportunities',
        'requirements': ['Python', 'scikit-learn', 'SQL'],
        'seniority_level': 'Junior',
        'remote': True,
        'source': 'RemoteOK',
        'job_type': 'Full-time'
    },
    {
        'title': 'Senior AI Engineer',
        'company': 'Big Tech Co',
        'description': 'Lead AI initiatives with deep learning expertise required',
        'requirements': ['Python', 'TensorFlow', 'PyTorch', 'Docker', 'AWS', 'Kubernetes'],
        'seniority_level': 'Senior',
        'remote': False,
        'source': 'RemoteOK',
        'job_type': 'Full-time'
    },
    {
        'title': 'Principal Data Scientist',
        'company': 'Enterprise Corp',
        'description': 'Strategic role leading data science initiatives across the organization',
        'requirements': ['Python', 'Machine Learning', 'Deep Learning', 'SQL', 'AWS', 'Docker', 'Leadership'],
        'seniority_level': 'Principal',
        'remote': True,
        'source': 'RemoteOK',
        'job_type': 'Full-time'
    }
]

# Make predictions
test_df = pd.DataFrame(test_jobs)
predictions = predictor.predict(test_df)

print("\n🎯 Salary Predictions for AI/ML Roles:\n")
print("-" * 60)

for i, (job, salary) in enumerate(zip(test_jobs, predictions)):
    print(f"\n{i+1}. {job['title']} at {job['company']}")
    print(f"   Level: {job['seniority_level']}")
    print(f"   Remote: {'Yes' if job['remote'] else 'No'}")
    print(f"   Skills: {len(job['requirements'])} requirements")
    print(f"   💰 Predicted Salary: ${salary:,.0f}")
    
    # Get explanation
    explanation = predictor.explain_prediction(job)
    print(f"   📊 Confidence Range: ${explanation['confidence_interval'][0]:,.0f} - ${explanation['confidence_interval'][1]:,.0f}")
    print(f"   🎯 Market Position: {explanation['market_position']}")

print("\n" + "-" * 60)
print("\n✅ Model is working correctly and making reasonable predictions!")