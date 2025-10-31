import { useEffect, useState } from 'react'

export default function Hero() {
  const [accuracyCount, setAccuracyCount] = useState(0)
  const [featuresCount, setFeaturesCount] = useState(0)
  const [modelsCount, setModelsCount] = useState(0)

  useEffect(() => {
    // Animate stats counting up
    const duration = 2000
    const steps = 60

    const accuracyIncrement = 92 / steps
    const featuresIncrement = 134 / steps
    const modelsIncrement = 3 / steps

    let currentStep = 0
    const interval = setInterval(() => {
      currentStep++
      if (currentStep <= steps) {
        setAccuracyCount(Math.min(92, Math.round(accuracyIncrement * currentStep)))
        setFeaturesCount(Math.min(134, Math.round(featuresIncrement * currentStep)))
        setModelsCount(Math.min(3, Math.round(modelsIncrement * currentStep)))
      } else {
        clearInterval(interval)
      }
    }, duration / steps)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="bg-gradient-to-br from-primary-600 to-primary-800 text-white">
      <div className="max-w-6xl mx-auto px-4 py-20">
        <div className="text-center">
          <h1 className="text-5xl md:text-6xl font-bold mb-6 animate-fade-in">
            JasperMatters
          </h1>
          <p className="text-xl md:text-2xl mb-4 text-primary-100">
            Live ML-Powered Job Market Intelligence
          </p>
          <p className="text-lg text-primary-200 max-w-3xl mx-auto mb-8">
            Production TensorFlow models for salary prediction, semantic job search,
            and career insights. Built by Matthew Scott, AI/ML Engineer.
          </p>

          {/* What's Real Disclaimer */}
          <div className="max-w-2xl mx-auto mb-12">
            <div className="bg-white/10 backdrop-blur-sm border border-white/20 rounded-lg p-4 text-sm">
              <p className="text-primary-100 mb-2">
                <strong className="text-white">💡 What's Real:</strong>
              </p>
              <div className="text-left text-primary-100 space-y-1">
                <p>✅ TensorFlow models trained and saved in GitHub repo</p>
                <p>✅ React app with real data visualization</p>
                <p>🔄 Demos use simulated predictions for instant response</p>
                <p>📅 Backend API integration coming soon</p>
              </div>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto mb-12">
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 transform transition-all duration-300 hover:scale-105">
              <div className="text-4xl font-bold mb-2">{accuracyCount}%</div>
              <div className="text-primary-100">Model Accuracy</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 transform transition-all duration-300 hover:scale-105">
              <div className="text-4xl font-bold mb-2">{featuresCount}</div>
              <div className="text-primary-100">Features Analyzed</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 transform transition-all duration-300 hover:scale-105">
              <div className="text-4xl font-bold mb-2">{modelsCount}</div>
              <div className="text-primary-100">Live ML Models</div>
            </div>
          </div>

          {/* CTA */}
          <div>
            <p className="text-primary-100 mb-4">Try the demos below ↓</p>
            <a
              href="https://github.com/guitargnarr/jaspermatters"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block bg-white text-primary-700 px-8 py-3 rounded-lg font-semibold hover:bg-primary-50 transition-all transform hover:scale-105"
            >
              View on GitHub
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}
