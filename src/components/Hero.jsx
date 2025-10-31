export default function Hero() {
  return (
    <div className="bg-gradient-to-br from-primary-600 to-primary-800 text-white">
      <div className="max-w-6xl mx-auto px-4 py-20">
        <div className="text-center">
          <h1 className="text-5xl md:text-6xl font-bold mb-6">
            JasperMatters
          </h1>
          <p className="text-xl md:text-2xl mb-4 text-primary-100">
            Live ML-Powered Job Market Intelligence
          </p>
          <p className="text-lg text-primary-200 max-w-3xl mx-auto mb-12">
            Production TensorFlow models for salary prediction, semantic job search,
            and career insights. Built by Matthew Scott, AI/ML Engineer.
          </p>

          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6">
              <div className="text-4xl font-bold mb-2">92%</div>
              <div className="text-primary-100">Model Accuracy</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6">
              <div className="text-4xl font-bold mb-2">134</div>
              <div className="text-primary-100">Features Analyzed</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6">
              <div className="text-4xl font-bold mb-2">3</div>
              <div className="text-primary-100">Live ML Models</div>
            </div>
          </div>

          {/* CTA */}
          <div className="mt-12">
            <p className="text-primary-100 mb-4">Try the demos below ↓</p>
            <a
              href="https://github.com/guitargnarr/jaspermatters"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block bg-white text-primary-700 px-8 py-3 rounded-lg font-semibold hover:bg-primary-50 transition-colors"
            >
              View on GitHub
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}
