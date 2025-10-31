export default function TechnicalDeepDive() {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">
          🔬 Technical Deep Dive
        </h2>
        <p className="text-gray-600">
          How the ML models work under the hood
        </p>
      </div>

      {/* Architecture */}
      <div className="card">
        <h3 className="text-2xl font-bold text-gray-900 mb-4">Model Architecture</h3>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
            <h4 className="font-semibold text-blue-900 mb-2">Salary Predictor</h4>
            <ul className="text-sm text-blue-800 space-y-2">
              <li>✓ TensorFlow/Keras neural network</li>
              <li>✓ 4 hidden layers (256→128→64→32)</li>
              <li>✓ Batch normalization & dropout</li>
              <li>✓ Adam optimizer (lr=0.001)</li>
              <li>✓ MSE loss function</li>
            </ul>
          </div>

          <div className="bg-purple-50 border border-purple-200 rounded-lg p-6">
            <h4 className="font-semibold text-purple-900 mb-2">Semantic Search</h4>
            <ul className="text-sm text-purple-800 space-y-2">
              <li>✓ Vector embeddings</li>
              <li>✓ TF-IDF text features</li>
              <li>✓ Cosine similarity matching</li>
              <li>✓ Real-time indexing</li>
              <li>✓ Sub-second queries</li>
            </ul>
          </div>

          <div className="bg-green-50 border border-green-200 rounded-lg p-6">
            <h4 className="font-semibold text-green-900 mb-2">Skill Analysis</h4>
            <ul className="text-sm text-green-800 space-y-2">
              <li>✓ NLP skill extraction</li>
              <li>✓ Multi-dimensional scoring</li>
              <li>✓ Market demand weighting</li>
              <li>✓ Gap prioritization</li>
              <li>✓ Learning pathways</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Features */}
      <div className="card">
        <h3 className="text-2xl font-bold text-gray-900 mb-4">Feature Engineering</h3>
        <div className="bg-gray-50 rounded-lg p-6">
          <h4 className="font-semibold text-gray-900 mb-3">134 Total Features</h4>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div>
              <h5 className="font-medium text-gray-900 mb-2">Categorical Features:</h5>
              <ul className="text-gray-700 space-y-1">
                <li>• Seniority level encoding</li>
                <li>• Source platform encoding</li>
                <li>• Job type encoding</li>
                <li>• Remote work indicator</li>
              </ul>
            </div>
            <div>
              <h5 className="font-medium text-gray-900 mb-2">Numerical Features:</h5>
              <ul className="text-gray-700 space-y-1">
                <li>• Number of requirements</li>
                <li>• Description length</li>
                <li>• Company name length</li>
                <li>• Years of experience</li>
              </ul>
            </div>
            <div>
              <h5 className="font-medium text-gray-900 mb-2">Skill Indicators:</h5>
              <ul className="text-gray-700 space-y-1">
                <li>• 14 key technical skills (binary)</li>
                <li>• ML framework presence</li>
                <li>• Cloud platform indicators</li>
                <li>• Programming languages</li>
              </ul>
            </div>
            <div>
              <h5 className="font-medium text-gray-900 mb-2">Text Features:</h5>
              <ul className="text-gray-700 space-y-1">
                <li>• TF-IDF vectors (100 features)</li>
                <li>• Title keyword indicators (13)</li>
                <li>• Description embeddings</li>
                <li>• Skill co-occurrence</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Training Data */}
      <div className="card">
        <h3 className="text-2xl font-bold text-gray-900 mb-4">Training & Performance</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold text-gray-900 mb-3">Training Process</h4>
            <ul className="text-sm text-gray-700 space-y-2">
              <li>📊 <strong>Dataset:</strong> Job market data from multiple sources</li>
              <li>🔄 <strong>Preprocessing:</strong> StandardScaler normalization</li>
              <li>✂️ <strong>Split:</strong> 80/20 train/validation</li>
              <li>⚡ <strong>Epochs:</strong> Early stopping with patience=20</li>
              <li>📉 <strong>Learning Rate:</strong> Adaptive with ReduceLROnPlateau</li>
              <li>💾 <strong>Checkpoints:</strong> Best weights restoration</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-gray-900 mb-3">Performance Metrics</h4>
            <div className="space-y-3">
              <div className="bg-green-50 border border-green-200 rounded p-3">
                <div className="text-2xl font-bold text-green-700">92%</div>
                <div className="text-sm text-green-800">Prediction Accuracy</div>
              </div>
              <div className="bg-blue-50 border border-blue-200 rounded p-3">
                <div className="text-2xl font-bold text-blue-700">$12K</div>
                <div className="text-sm text-blue-800">Mean Absolute Error</div>
              </div>
              <div className="bg-purple-50 border border-purple-200 rounded p-3">
                <div className="text-2xl font-bold text-purple-700">&lt;100ms</div>
                <div className="text-sm text-purple-800">Inference Time</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Tech Stack */}
      <div className="card">
        <h3 className="text-2xl font-bold text-gray-900 mb-4">Technology Stack</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-3xl mb-2">🐍</div>
            <div className="font-semibold">Python 3.9+</div>
            <div className="text-sm text-gray-600">Core Language</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-3xl mb-2">🧠</div>
            <div className="font-semibold">TensorFlow</div>
            <div className="text-sm text-gray-600">Deep Learning</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-3xl mb-2">📊</div>
            <div className="font-semibold">Pandas</div>
            <div className="text-sm text-gray-600">Data Processing</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-3xl mb-2">🔬</div>
            <div className="font-semibold">Scikit-learn</div>
            <div className="text-sm text-gray-600">ML Tools</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-3xl mb-2">⚡</div>
            <div className="font-semibold">FastAPI</div>
            <div className="text-sm text-gray-600">API Server</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-3xl mb-2">⚛️</div>
            <div className="font-semibold">React</div>
            <div className="text-sm text-gray-600">Frontend</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-3xl mb-2">🎨</div>
            <div className="font-semibold">Tailwind</div>
            <div className="text-sm text-gray-600">Styling</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-3xl mb-2">🚀</div>
            <div className="font-semibold">Netlify</div>
            <div className="text-sm text-gray-600">Deployment</div>
          </div>
        </div>
      </div>

      {/* GitHub Link */}
      <div className="card bg-gradient-to-r from-gray-900 to-gray-800 text-white">
        <div className="text-center">
          <h3 className="text-2xl font-bold mb-4">View the Source Code</h3>
          <p className="text-gray-300 mb-6">
            All models, training code, and infrastructure are open source on GitHub
          </p>
          <a
            href="https://github.com/guitargnarr/jaspermatters"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 bg-white text-gray-900 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
          >
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
              <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
            </svg>
            guitargnarr/jaspermatters
          </a>
        </div>
      </div>
    </div>
  )
}
