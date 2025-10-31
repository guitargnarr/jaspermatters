/**
 * E2E tests for JasperMatters ML Platform
 * Run with: npx playwright test
 */

const { test, expect } = require('@playwright/test')

test.describe('JasperMatters ML Platform', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('https://jaspermatters.com')
  })

  test('homepage loads correctly', async ({ page }) => {
    await expect(page).toHaveTitle(/JasperMatters/)
    await expect(page.locator('h1')).toContainText('JasperMatters')
  })

  test('navigation pills are visible', async ({ page }) => {
    await expect(page.getByRole('button', { name: /Salary Predictor/i })).toBeVisible()
    await expect(page.getByRole('button', { name: /Job Search/i })).toBeVisible()
    await expect(page.getByRole('button', { name: /Skill Gap/i })).toBeVisible()
    await expect(page.getByRole('button', { name: /Technical Details/i })).toBeVisible()
  })

  test('salary predictor demo works', async ({ page }) => {
    // Click Salary Predictor (should be active by default)
    await expect(page.getByText('AI Salary Predictor')).toBeVisible()

    // Click "Load Example" button
    await page.getByRole('button', { name: /Load Example/i }).first().click()

    // Verify form is filled
    await expect(page.locator('input[type="text"]').first()).toHaveValue(/Machine Learning/)

    // Submit form
    await page.getByRole('button', { name: /Predict Salary/i }).click()

    // Wait for loading to finish and results to appear
    await page.waitForTimeout(2000)

    // Check for results
    await expect(page.getByText(/Salary Prediction/i)).toBeVisible()
  })

  test('job search demo works', async ({ page }) => {
    // Switch to Job Search tab
    await page.getByRole('button', { name: /Job Search/i }).click()

    // Wait for component to load
    await expect(page.getByText('Semantic Job Search')).toBeVisible()

    // Click "Load Example"
    await page.getByRole('button', { name: /Load Example/i }).click()

    // Submit search
    await page.getByRole('button', { name: /Search Jobs/i }).click()

    // Wait for results
    await page.waitForTimeout(1500)

    // Check for job cards
    await expect(page.getByText(/matching jobs/i)).toBeVisible()
  })

  test('skill gap analyzer works', async ({ page }) => {
    // Switch to Skill Gap tab
    await page.getByRole('button', { name: /Skill Gap/i }).click()

    // Wait for component
    await expect(page.getByText('Skill Gap Analyzer')).toBeVisible()

    // Load example
    await page.getByRole('button', { name: /Load Example/i }).click()

    // Analyze
    await page.getByRole('button', { name: /Analyze Skills/i }).click()

    // Wait for results
    await page.waitForTimeout(2000)

    // Check for match score
    await expect(page.getByText(/Match Score/i)).toBeVisible()
  })

  test('technical deep dive section loads', async ({ page }) => {
    // Switch to Technical Details tab
    await page.getByRole('button', { name: /Technical Details/i }).click()

    // Check content loads
    await expect(page.getByText(/Technical Deep Dive/i)).toBeVisible()
    await expect(page.getByText(/Model Architecture/i)).toBeVisible()
    await expect(page.getByText(/134 Total Features/i)).toBeVisible()
  })

  test('footer contains contact information', async ({ page }) => {
    await expect(page.getByText('matthewdscott7@gmail.com')).toBeVisible()
    await expect(page.getByText(/Matthew Scott/i)).toBeVisible()
  })

  test('GitHub link works', async ({ page }) => {
    const githubLink = page.getByRole('link', { name: /View on GitHub/i }).first()
    await expect(githubLink).toHaveAttribute('href', /github\.com/)
  })

  test('mobile responsive - navigation scrolls', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 })
    const navContainer = page.locator('.overflow-x-auto').first()
    await expect(navContainer).toBeVisible()
  })
})
