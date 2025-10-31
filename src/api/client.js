/**
 * API Client for JasperMatters ML Platform
 * Handles all backend API calls with fallback to mock data
 */

import axios from 'axios'

// API base URL - will be set via environment variable
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Flag to use real API or mock data
const USE_REAL_API = import.meta.env.VITE_USE_REAL_API === 'true'

class MLAPIClient {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json'
      }
    })
  }

  /**
   * Predict salary for a job posting
   */
  async predictSalary(formData) {
    if (!USE_REAL_API) {
      return this.mockPredictSalary(formData)
    }

    try {
      const response = await this.client.post('/api/predict-salary', {
        title: formData.title,
        seniority: formData.seniority,
        remote: formData.remote,
        yearsExp: parseInt(formData.yearsExp),
        skills: formData.skills
      })
      return response.data
    } catch (error) {
      console.error('API error, falling back to mock:', error)
      return this.mockPredictSalary(formData)
    }
  }

  /**
   * Search jobs using semantic search
   */
  async searchJobs(query, topK = 5) {
    if (!USE_REAL_API) {
      return this.mockSearchJobs(query, topK)
    }

    try {
      const response = await this.client.post('/api/search-jobs', {
        query,
        top_k: topK
      })
      return response.data
    } catch (error) {
      console.error('API error, falling back to mock:', error)
      return this.mockSearchJobs(query, topK)
    }
  }

  /**
   * Analyze skill gaps
   */
  async analyzeSkills(resume, targetRole) {
    if (!USE_REAL_API) {
      return this.mockAnalyzeSkills(resume, targetRole)
    }

    try {
      const response = await this.client.post('/api/analyze-skills', {
        resume,
        target_role: targetRole
      })
      return response.data
    } catch (error) {
      console.error('API error, falling back to mock:', error)
      return this.mockAnalyzeSkills(resume, targetRole)
    }
  }

  // Mock implementations (fallback when API unavailable)

  mockPredictSalary(formData) {
    return new Promise((resolve) => {
      setTimeout(() => {
        const baseSalaries = {
          'Junior': 80000,
          'Mid-level': 110000,
          'Senior': 150000,
          'Lead': 180000,
          'Principal': 210000
        }

        const base = baseSalaries[formData.seniority]
        const skillBonus = formData.skills.length * 5000
        const remoteBonus = formData.remote ? 10000 : 0
        const predicted = base + skillBonus + remoteBonus + (Math.random() - 0.5) * 20000

        resolve({
          predicted_salary: predicted,
          confidence_range: [predicted * 0.85, predicted * 1.15],
          factors: {
            seniority: formData.seniority,
            skill_count: formData.skills.length,
            is_remote: formData.remote,
            top_skills: formData.skills.slice(0, 3)
          },
          market_position: predicted > 140000 ? 'Above Average' : 'Average',
          model_info: {
            model_type: "TensorFlow Neural Network (Mock)",
            features: 134,
            accuracy: "92%",
            inference_time_ms: "<100"
          }
        })
      }, 1500)
    })
  }

  mockSearchJobs(query, topK) {
    return new Promise((resolve) => {
      setTimeout(() => {
        const mockJobs = [
          {
            id: 1,
            title: 'Senior Machine Learning Engineer',
            company: 'TechCorp AI',
            location: 'San Francisco, CA',
            remote: true,
            salary_range: [160000, 200000],
            description: 'Build production ML systems at scale. Work with TensorFlow and PyTorch.',
            skills: ['Python', 'TensorFlow', 'PyTorch', 'Docker', 'Kubernetes'],
            seniority: 'Senior',
            score: 0.95
          },
          {
            id: 2,
            title: 'ML Engineer - Computer Vision',
            company: 'Vision Labs',
            location: 'Remote',
            remote: true,
            salary_range: [140000, 180000],
            description: 'Join our computer vision team building real-time object detection systems.',
            skills: ['Python', 'PyTorch', 'OpenCV', 'CUDA', 'AWS'],
            seniority: 'Mid-Senior',
            score: 0.88
          },
          {
            id: 3,
            title: 'Staff ML Platform Engineer',
            company: 'DataScale Inc',
            location: 'New York, NY',
            remote: false,
            salary_range: [180000, 220000],
            description: 'Lead ML infrastructure development. Design and implement MLOps platforms.',
            skills: ['Python', 'Kubernetes', 'Terraform', 'AWS', 'MLflow'],
            seniority: 'Staff',
            score: 0.82
          }
        ]

        const keywords = query.toLowerCase().split(' ')
        const scoredJobs = mockJobs.map(job => {
          const text = `${job.title} ${job.description} ${job.skills.join(' ')}`.toLowerCase()
          const matches = keywords.filter(keyword => text.includes(keyword)).length
          const score = matches / keywords.length

          return {
            ...job,
            score: score > 0 ? score * 0.9 + Math.random() * 0.1 : Math.random() * 0.3
          }
        }).sort((a, b) => b.score - a.score)

        resolve({
          jobs: scoredJobs.slice(0, topK),
          total_found: scoredJobs.length,
          query
        })
      }, 1200)
    })
  }

  mockAnalyzeSkills(resume, targetRole) {
    return new Promise((resolve) => {
      setTimeout(() => {
        const roleRequirements = {
          'Senior ML Engineer': ['Python', 'TensorFlow', 'PyTorch', 'Docker', 'Kubernetes', 'AWS', 'SQL'],
          'Data Scientist': ['Python', 'Pandas', 'Scikit-learn', 'SQL', 'Statistics', 'Visualization'],
          'ML Platform Engineer': ['Python', 'Kubernetes', 'Terraform', 'MLflow', 'CI/CD', 'Cloud'],
          'Computer Vision Engineer': ['Python', 'PyTorch', 'OpenCV', 'CUDA', 'Deep Learning']
        }

        const requiredSkills = roleRequirements[targetRole] || []
        const resumeLower = resume.toLowerCase()

        const allSkills = Object.values(roleRequirements).flat()
        const uniqueSkills = [...new Set(allSkills)]
        const foundSkills = uniqueSkills.filter(skill => resumeLower.includes(skill.toLowerCase()))

        const matchingSkills = requiredSkills.filter(skill =>
          foundSkills.some(f => f.toLowerCase() === skill.toLowerCase())
        )
        const missingSkills = requiredSkills.filter(skill =>
          !foundSkills.some(f => f.toLowerCase() === skill.toLowerCase())
        )

        const matchPercentage = (matchingSkills.length / requiredSkills.length) * 100

        const recommendation =
          matchPercentage >= 70 ? 'Strong match! Apply now.' :
          matchPercentage >= 50 ? 'Good match. Consider upskilling in missing areas.' :
          'Significant gaps. Focus on learning priority skills first.'

        const prioritySkills = missingSkills.slice(0, 5).map(skill => ({
          skill,
          demand: 75 + Math.floor(Math.random() * 25)
        }))

        resolve({
          matching_skills: matchingSkills,
          missing_skills: missingSkills,
          match_percentage: matchPercentage,
          recommendation,
          priority_skills: prioritySkills
        })
      }, 1500)
    })
  }
}

// Export singleton instance
export const apiClient = new MLAPIClient()
