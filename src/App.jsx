import { useState } from 'react'
import Hero from './components/Hero'
import SalaryPredictor from './components/SalaryPredictor'
import JobSearch from './components/JobSearch'
import SkillGapAnalyzer from './components/SkillGapAnalyzer'
import TechnicalDeepDive from './components/TechnicalDeepDive'
import Footer from './components/Footer'

function App() {
  const [activeDemo, setActiveDemo] = useState('salary')

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <Hero />

      {/* Navigation Pills */}
      <div className="bg-white border-b sticky top-0 z-10 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex gap-2 overflow-x-auto">
            <button
              onClick={() => setActiveDemo('salary')}
              className={`px-6 py-2 rounded-full font-medium whitespace-nowrap transition-colors ${
                activeDemo === 'salary'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              💰 Salary Predictor
            </button>
            <button
              onClick={() => setActiveDemo('search')}
              className={`px-6 py-2 rounded-full font-medium whitespace-nowrap transition-colors ${
                activeDemo === 'search'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              🔍 Job Search
            </button>
            <button
              onClick={() => setActiveDemo('skills')}
              className={`px-6 py-2 rounded-full font-medium whitespace-nowrap transition-colors ${
                activeDemo === 'skills'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              📊 Skill Gap
            </button>
            <button
              onClick={() => setActiveDemo('technical')}
              className={`px-6 py-2 rounded-full font-medium whitespace-nowrap transition-colors ${
                activeDemo === 'technical'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              🔬 Technical Details
            </button>
          </div>
        </div>
      </div>

      {/* Demo Sections */}
      <div className="max-w-6xl mx-auto px-4 py-12">
        {activeDemo === 'salary' && <SalaryPredictor />}
        {activeDemo === 'search' && <JobSearch />}
        {activeDemo === 'skills' && <SkillGapAnalyzer />}
        {activeDemo === 'technical' && <TechnicalDeepDive />}
      </div>

      {/* Footer */}
      <Footer />
    </div>
  )
}

export default App
