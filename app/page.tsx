'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'

export default function HomePage() {
  const [url, setUrl] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [loadingStep, setLoadingStep] = useState(0)
  const router = useRouter()

  const loadingMessages = [
    'Reading your website...',
    'Building your AI profile...',
    'Almost done...',
  ]

  function validateUrl(value: string): boolean {
    try {
      const parsed = new URL(value)
      return ['http:', 'https:'].includes(parsed.protocol)
    } catch {
      return false
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')

    const trimmed = url.trim()
    const withProtocol = trimmed.startsWith('http') ? trimmed : `https://${trimmed}`

    if (!validateUrl(withProtocol)) {
      setError('Please enter a valid website URL.')
      return
    }

    setLoading(true)
    setLoadingStep(0)

    const interval = setInterval(() => {
      setLoadingStep(prev => Math.min(prev + 1, loadingMessages.length - 1))
    }, 2500)

    try {
      const res = await fetch('/api/onboard', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: withProtocol }),
      })

      const data = await res.json()

      if (!res.ok) {
        setError(data.error || 'Something went wrong. Please try again.')
        return
      }

      router.push(`/setup?id=${data.business_id}`)
    } catch {
      setError('Network error. Please try again.')
    } finally {
      clearInterval(interval)
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-white flex flex-col">
      {/* Header */}
      <header className="px-6 py-5 flex items-center justify-between max-w-5xl mx-auto w-full">
        <span className="text-xl font-bold tracking-tight">Meshly</span>
        <a
          href="/login"
          className="text-sm text-gray-500 hover:text-gray-900 transition-colors"
        >
          Sign in
        </a>
      </header>

      {/* Main content */}
      <main className="flex-1 flex flex-col items-center justify-center px-6 py-16">
        <div className="w-full max-w-2xl text-center">
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-gray-900 mb-4 leading-tight">
            Make your business visible to AI.
            <br />
            Get leads automatically.
          </h1>
          <p className="text-lg text-gray-500 mb-10">
            Free. No credit card. 30 seconds.
          </p>

          {loading ? (
            <div className="py-8">
              <div className="flex items-center justify-center gap-3 mb-4">
                <div className="w-5 h-5 border-2 border-gray-900 border-t-transparent rounded-full animate-spin" />
                <span className="text-gray-700 font-medium text-lg">
                  {loadingMessages[loadingStep]}
                </span>
              </div>
              <div className="flex justify-center gap-2">
                {loadingMessages.map((_, i) => (
                  <div
                    key={i}
                    className={`h-1.5 w-8 rounded-full transition-colors duration-500 ${
                      i <= loadingStep ? 'bg-gray-900' : 'bg-gray-200'
                    }`}
                  />
                ))}
              </div>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="w-full">
              <div className="flex flex-col sm:flex-row gap-3 w-full">
                <input
                  type="text"
                  value={url}
                  onChange={e => setUrl(e.target.value)}
                  placeholder="https://your-website.com"
                  className="flex-1 px-5 py-4 text-base border-2 border-gray-200 rounded-xl focus:outline-none focus:border-gray-900 transition-colors placeholder-gray-400"
                  autoFocus
                />
                <button
                  type="submit"
                  className="px-7 py-4 bg-gray-900 text-white font-semibold rounded-xl hover:bg-gray-700 transition-colors whitespace-nowrap text-base"
                >
                  Generate my AI profile →
                </button>
              </div>
              {error && (
                <p className="mt-3 text-sm text-red-500 text-left">{error}</p>
              )}
            </form>
          )}
        </div>

        {/* How it works */}
        <div className="mt-24 w-full max-w-3xl">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-8 text-center">
            {[
              {
                step: '①',
                title: 'We read your website',
                desc: 'We analyze your site and extract what you do, where you are, and who you serve.',
              },
              {
                step: '②',
                title: 'We optimize it for AI',
                desc: 'We generate a structured profile that AI assistants like ChatGPT and Claude can read.',
              },
              {
                step: '③',
                title: 'We send you relevant contacts',
                desc: 'When we find a business in our network that matches your needs, we email you.',
              },
            ].map(item => (
              <div key={item.step} className="flex flex-col items-center">
                <span className="text-3xl mb-4">{item.step}</span>
                <h3 className="font-semibold text-gray-900 mb-2">{item.title}</h3>
                <p className="text-sm text-gray-500 leading-relaxed">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </main>

      <footer className="py-6 text-center text-sm text-gray-400">
        © {new Date().getFullYear()} Meshly. Free forever.
      </footer>
    </div>
  )
}
