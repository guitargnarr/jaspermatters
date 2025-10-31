import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import Hero from '../Hero'

describe('Hero Component', () => {
  it('renders main heading', () => {
    render(<Hero />)
    expect(screen.getByText('JasperMatters')).toBeInTheDocument()
  })

  it('shows subtitle', () => {
    render(<Hero />)
    expect(screen.getByText(/Live ML-Powered Job Market Intelligence/i)).toBeInTheDocument()
  })

  it('displays author attribution', () => {
    render(<Hero />)
    expect(screen.getByText(/Matthew Scott, AI\/ML Engineer/i)).toBeInTheDocument()
  })

  it('shows transparency disclaimer', () => {
    render(<Hero />)
    expect(screen.getByText(/What's Real:/i)).toBeInTheDocument()
  })

  it('has GitHub link', () => {
    render(<Hero />)
    const githubLink = screen.getByRole('link', { name: /View on GitHub/i })
    expect(githubLink).toHaveAttribute('href', 'https://github.com/guitargnarr/jaspermatters')
  })

  it('displays stats cards', () => {
    render(<Hero />)
    expect(screen.getByText('Model Accuracy')).toBeInTheDocument()
    expect(screen.getByText('Features Analyzed')).toBeInTheDocument()
    expect(screen.getByText('Live ML Models')).toBeInTheDocument()
  })
})
