import { useState, lazy, Suspense } from 'react'
import Hero from './components/Hero'
import Footer from './components/Footer'

// Lazy load demo components for better performance
const SalaryPredictor = lazy(() => import('./components/SalaryPredictor'))
const JobSearch = lazy(() => import('./components/JobSearch'))
const SkillGapAnalyzer = lazy(() => import('./components/SkillGapAnalyzer'))
const TechnicalDeepDive = lazy(() => import('./components/TechnicalDeepDive'))

// Loading spinner component
const LoadingSpinner = () => (
  <div className="flex items-center justify-center py-20">
    <div className="text-center">
      <svg className="animate-spin h-12 w-12 text-primary-600 mx-auto mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
      </svg>
      <p className="text-gray-600">Loading demo...</p>
    </div>
  </div>
)

function App() {
  const [activeDemo, setActiveDemo] = useState('salary')

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <Hero />

      {/* Navigation Pills */}
      <div className="bg-white border-b sticky top-0 z-10 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex gap-2 overflow-x-auto pb-2 md:pb-0">
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

      {/* Demo Sections with Lazy Loading */}
      <div className="max-w-6xl mx-auto px-4 py-12">
        <Suspense fallback={<LoadingSpinner />}>
          {activeDemo === 'salary' && <SalaryPredictor />}
          {activeDemo === 'search' && <JobSearch />}
          {activeDemo === 'skills' && <SkillGapAnalyzer />}
          {activeDemo === 'technical' && <TechnicalDeepDive />}
        </Suspense>
      </div>

      {/* Footer */}
      <Footer />
    </div>
  )
}

export default App
