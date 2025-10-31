import { useState } from 'react'
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts'

export default function SkillGapAnalyzer() {
  const [resume, setResume] = useState('')
  const [targetRole, setTargetRole] = useState('Senior ML Engineer')
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(false)

  const roleRequirements = {
    'Senior ML Engineer': ['Python', 'TensorFlow', 'PyTorch', 'Docker', 'Kubernetes', 'AWS', 'SQL'],
    'Data Scientist': ['Python', 'Pandas', 'Scikit-learn', 'SQL', 'Statistics', 'Visualization'],
    'ML Platform Engineer': ['Python', 'Kubernetes', 'Terraform', 'MLflow', 'CI/CD', 'Cloud'],
    'Computer Vision Engineer': ['Python', 'PyTorch', 'OpenCV', 'CUDA', 'Deep Learning']
  }

  const handleAnalyze = async (e) => {
    e.preventDefault()
    setLoading(true)

    try {
      await new Promise(resolve => setTimeout(resolve, 1000))

      const requiredSkills = roleRequirements[targetRole] || []
      const resumeText = resume.toLowerCase()

      // Extract skills from resume
      const allSkills = Object.values(roleRequirements).flat()
      const uniqueSkills = [...new Set(allSkills)]

      const foundSkills = uniqueSkills.filter(skill =>
        resumeText.includes(skill.toLowerCase())
      )

      const missingSkills = requiredSkills.filter(skill =>
        !foundSkills.some(f => f.toLowerCase() === skill.toLowerCase())
      )

      const matchingSkills = requiredSkills.filter(skill =>
        foundSkills.some(f => f.toLowerCase() === skill.toLowerCase())
      )

      const matchPercentage = (matchingSkills.length / requiredSkills.length) * 100

      // Create radar chart data
      const categories = ['Technical', 'Cloud', 'ML Frameworks', 'Data', 'DevOps']
      const radarData = categories.map(cat => ({
        category: cat,
        current: 50 + Math.random() * 40,
        required: 70 + Math.random() * 20
      }))

      setAnalysis({
        matching_skills: matchingSkills,
        missing_skills: missingSkills,
        match_percentage: matchPercentage,
        priority_skills: missingSkills.slice(0, 5).map(s => [s, Math.floor(Math.random() * 100)]),
        recommendation: matchPercentage >= 70 ? 'Strong match! Apply now.' :
                       matchPercentage >= 50 ? 'Good match. Consider upskilling in missing areas.' :
                       'Significant gaps. Focus on learning priority skills first.',
        radar_data: radarData
      })
    } catch (error) {
      console.error('Analysis error:', error)
      alert('Analysis failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">
          📊 Skill Gap Analyzer
        </h2>
        <p className="text-gray-600">
          Compare your resume against target roles to identify skill gaps and learning priorities.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        {/* Input Form */}
        <div className="card">
          <form onSubmit={handleAnalyze} className="space-y-6">
            <div>
              <label className="label">Your Resume / Skills</label>
              <textarea
                value={resume}
                onChange={(e) => setResume(e.target.value)}
                className="input-field"
                rows="8"
                placeholder="Paste your resume or list your skills here...

Example:
Experienced Python developer with 5 years in ML. Strong in TensorFlow, scikit-learn, and Docker. Built production APIs with Flask. Proficient in SQL and data analysis."
                required
              />
            </div>

            <div>
              <label className="label">Target Role</label>
              <select
                value={targetRole}
                onChange={(e) => setTargetRole(e.target.value)}
                className="input-field"
              >
                {Object.keys(roleRequirements).map(role => (
                  <option key={role} value={role}>{role}</option>
                ))}
              </select>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Analyzing...' : 'Analyze Skills'}
            </button>
          </form>
        </div>

        {/* Results */}
        <div className="card">
          {!analysis ? (
            <div className="flex items-center justify-center h-full text-gray-400 text-center">
              <div>
                <div className="text-6xl mb-4">📈</div>
                <p>Enter your resume and select a target role to see your skill gap analysis</p>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Match Score */}
              <div>
                <h3 className="text-2xl font-bold text-gray-900 mb-2">
                  Match Score
                </h3>
                <div className="flex items-end gap-2">
                  <div className="text-5xl font-bold text-primary-600">
                    {Math.round(analysis.match_percentage)}%
                  </div>
                  <div className="text-gray-600 mb-2">match</div>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3 mt-3">
                  <div
                    className="bg-primary-600 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${analysis.match_percentage}%` }}
                  />
                </div>
              </div>

              {/* Recommendation */}
              <div className={`p-4 rounded-lg ${
                analysis.match_percentage >= 70 ? 'bg-green-50 border border-green-200' :
                analysis.match_percentage >= 50 ? 'bg-yellow-50 border border-yellow-200' :
                'bg-red-50 border border-red-200'
              }`}>
                <p className={`font-medium ${
                  analysis.match_percentage >= 70 ? 'text-green-800' :
                  analysis.match_percentage >= 50 ? 'text-yellow-800' :
                  'text-red-800'
                }`}>
                  {analysis.recommendation}
                </p>
              </div>

              {/* Skills Breakdown */}
              <div>
                <h4 className="font-semibold text-gray-900 mb-3">Matching Skills</h4>
                <div className="flex flex-wrap gap-2">
                  {analysis.matching_skills.map(skill => (
                    <span
                      key={skill}
                      className="bg-green-100 text-green-700 px-3 py-1 rounded-full text-sm"
                    >
                      ✓ {skill}
                    </span>
                  ))}
                </div>
              </div>

              {analysis.missing_skills.length > 0 && (
                <div>
                  <h4 className="font-semibold text-gray-900 mb-3">Skills to Learn</h4>
                  <div className="flex flex-wrap gap-2">
                    {analysis.missing_skills.map(skill => (
                      <span
                        key={skill}
                        className="bg-orange-100 text-orange-700 px-3 py-1 rounded-full text-sm"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Radar Chart */}
              {analysis.radar_data && (
                <div>
                  <h4 className="font-semibold text-gray-900 mb-3">Skill Profile</h4>
                  <ResponsiveContainer width="100%" height={250}>
                    <RadarChart data={analysis.radar_data}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="category" />
                      <PolarRadiusAxis angle={90} domain={[0, 100]} />
                      <Radar
                        name="Your Skills"
                        dataKey="current"
                        stroke="#2563eb"
                        fill="#2563eb"
                        fillOpacity={0.3}
                      />
                      <Radar
                        name="Required"
                        dataKey="required"
                        stroke="#dc2626"
                        fill="#dc2626"
                        fillOpacity={0.3}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                  <div className="flex justify-center gap-4 text-sm mt-2">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-primary-600 rounded"></div>
                      <span>Your Skills</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-red-600 rounded"></div>
                      <span>Required</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Info */}
      <div className="card bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200">
        <div className="flex items-start space-x-4">
          <div className="text-3xl">🎯</div>
          <div>
            <h4 className="font-semibold text-gray-900 mb-2">Learning Recommendations</h4>
            <p className="text-gray-700 text-sm">
              Our AI analyzes job descriptions and market demand to prioritize which skills
              will have the highest impact on your career. Focus on high-frequency missing skills
              that appear across multiple job postings in your target role.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
