export default function Footer() {
  return (
    <footer className="bg-gray-900 text-white mt-20">
      <div className="max-w-6xl mx-auto px-4 py-12">
        <div className="grid md:grid-cols-3 gap-8">
          {/* About */}
          <div>
            <h3 className="text-xl font-bold mb-4">About This Project</h3>
            <p className="text-gray-400 text-sm mb-4">
              JasperMatters is a production ML platform demonstrating real-world
              applications of machine learning in job market intelligence.
            </p>
            <p className="text-gray-400 text-sm">
              Built with TensorFlow, React, and deployed on Netlify.
            </p>
          </div>

          {/* Links */}
          <div>
            <h3 className="text-xl font-bold mb-4">Links</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a
                  href="https://github.com/guitargnarr/jaspermatters"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  GitHub Repository
                </a>
              </li>
              <li>
                <a
                  href="https://interactive-resume-ten-pi.vercel.app"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  My Full Portfolio
                </a>
              </li>
              <li>
                <a
                  href="https://linkedin.com/in/mscott77"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  LinkedIn
                </a>
              </li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h3 className="text-xl font-bold mb-4">Contact</h3>
            <div className="space-y-2 text-sm text-gray-400">
              <div>
                <strong className="text-white">Matthew Scott</strong>
              </div>
              <div>AI/ML Engineer</div>
              <div>Louisville, KY</div>
              <div>
                <a
                  href="mailto:matthewdscott7@gmail.com"
                  className="hover:text-white transition-colors"
                >
                  matthewdscott7@gmail.com
                </a>
              </div>
              <div>
                <a
                  href="tel:502-345-0525"
                  className="hover:text-white transition-colors"
                >
                  (502) 345-0525
                </a>
              </div>
            </div>
          </div>
        </div>

        <div className="border-t border-gray-800 mt-8 pt-8 text-center text-sm text-gray-400">
          <p>
            © 2025 Matthew Scott. Built with React + TensorFlow.
            All ML models are for demonstration purposes.
          </p>
        </div>
      </div>
    </footer>
  )
}
