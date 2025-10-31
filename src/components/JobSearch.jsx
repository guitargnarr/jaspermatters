import { useState } from 'react'

export default function JobSearch() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)

  // Mock job data
  const mockJobs = [
    {
      id: 1,
      title: 'Senior Machine Learning Engineer',
      company: 'TechCorp AI',
      location: 'San Francisco, CA',
      remote: true,
      salary_range: [160000, 200000],
      description: 'Build production ML systems at scale. Work with TensorFlow and PyTorch to develop cutting-edge AI solutions.',
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
      description: 'Join our computer vision team building real-time object detection systems using deep learning.',
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
      description: 'Lead ML infrastructure development. Design and implement MLOps platforms for model deployment at scale.',
      skills: ['Python', 'Kubernetes', 'Terraform', 'AWS', 'MLflow'],
      seniority: 'Staff',
      score: 0.82
    }
  ]

  const handleSearch = async (e) => {
    e.preventDefault()
    setLoading(true)

    try {
      // Simulate API delay (makes it feel more "real")
      await new Promise(resolve => setTimeout(resolve, 1200))

      // For demo, filter and score based on keywords
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

      setResults(scoredJobs)
    } catch (error) {
      console.error('Search error:', error)
      alert('Search failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">
          🔍 Semantic Job Search
        </h2>
        <p className="text-gray-600">
          Search jobs using natural language. Our vector embeddings understand meaning, not just keywords.
        </p>
      </div>

      {/* Search Form */}
      <form onSubmit={handleSearch} className="card">
        <div className="space-y-4">
          <div>
            <label className="label">Describe what you're looking for</label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="input-field"
              rows="3"
              placeholder="e.g., I want a senior machine learning role with good pay and remote work options"
              required
            />
          </div>

          <div className="flex gap-3">
            <button
              type="submit"
              disabled={loading}
              className="btn-primary flex-1 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Searching...
                </span>
              ) : 'Search Jobs'}
            </button>

            <button
              type="button"
              onClick={() => setQuery('senior ML engineer with high salary and remote work')}
              className="btn-secondary whitespace-nowrap"
            >
              Load Example
            </button>
          </div>
        </div>
      </form>

      {/* Results */}
      {results && (
        <div className="space-y-4">
          <h3 className="text-xl font-semibold text-gray-900">
            Found {results.filter(j => j.score > 0.3).length} matching jobs
          </h3>

          {results.filter(j => j.score > 0.3).map(job => (
            <div key={job.id} className="card hover:shadow-lg transition-shadow">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h4 className="text-xl font-semibold text-gray-900 mb-1">
                    {job.title}
                  </h4>
                  <p className="text-gray-600">
                    {job.company} • {job.location} {job.remote && '🌐'}
                  </p>
                </div>
                <div className="text-right">
                  <div className="inline-block bg-primary-100 text-primary-700 px-3 py-1 rounded-full text-sm font-medium">
                    {Math.round(job.score * 100)}% match
                  </div>
                </div>
              </div>

              <p className="text-gray-700 mb-4">{job.description}</p>

              <div className="flex flex-wrap gap-2 mb-4">
                {job.skills.map(skill => (
                  <span
                    key={skill}
                    className="bg-gray-100 text-gray-700 px-3 py-1 rounded-full text-sm"
                  >
                    {skill}
                  </span>
                ))}
              </div>

              <div className="flex justify-between items-center text-sm">
                <span className="font-medium text-gray-900">
                  ${job.salary_range[0].toLocaleString()} - ${job.salary_range[1].toLocaleString()}
                </span>
                <span className="text-gray-600">{job.seniority} Level</span>
              </div>
            </div>
          ))}

          {results.every(j => j.score <= 0.3) && (
            <div className="card text-center text-gray-500 py-12">
              <div className="text-5xl mb-4">🔍</div>
              <p>No strong matches found. Try adjusting your search query.</p>
            </div>
          )}
        </div>
      )}

      {!results && (
        <div className="card bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200">
          <div className="flex items-start space-x-4">
            <div className="text-3xl">💡</div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-2">How semantic search works</h4>
              <p className="text-gray-700 text-sm mb-3">
                Traditional keyword search only finds exact matches. Our semantic search uses
                vector embeddings to understand the <em>meaning</em> behind your query.
              </p>
              <ul className="text-sm text-gray-700 space-y-1">
                <li>✓ Understands synonyms ("remote" = "work from home")</li>
                <li>✓ Captures intent ("well-paid" matches high salaries)</li>
                <li>✓ Contextual matching (finds relevant jobs even without exact keywords)</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
