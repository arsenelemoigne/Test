'use client'

import { useState, useEffect, Suspense } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'

function SetupContent() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const businessId = searchParams.get('id')

  const [business, setBusiness] = useState<{
    name: string
    llms_txt_content: string
  } | null>(null)
  const [email, setEmail] = useState('')
  const [agreed, setAgreed] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [fetching, setFetching] = useState(true)

  useEffect(() => {
    if (!businessId) {
      router.push('/')
      return
    }

    fetch(`/api/business/${businessId}`)
      .then(r => r.json())
      .then(data => {
        if (data.error) {
          router.push('/')
          return
        }
        setBusiness(data)
      })
      .catch(() => router.push('/'))
      .finally(() => setFetching(false))
  }, [businessId, router])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')

    if (!agreed) {
      setError('Please agree to receive B2B contact opportunities.')
      return
    }

    setLoading(true)

    try {
      const res = await fetch('/api/activate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ business_id: businessId, email }),
      })

      const data = await res.json()

      if (!res.ok) {
        setError(data.error || 'Something went wrong.')
        return
      }

      router.push('/thanks')
    } catch {
      setError('Network error. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  if (fetching) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="w-6 h-6 border-2 border-gray-900 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  if (!business) return null

  return (
    <div className="min-h-screen bg-white flex flex-col">
      <header className="px-6 py-5 max-w-5xl mx-auto w-full">
        <span className="text-xl font-bold tracking-tight">Meshly</span>
      </header>

      <main className="flex-1 max-w-2xl mx-auto w-full px-6 py-12">
        <div className="mb-8">
          <div className="inline-flex items-center gap-2 text-sm text-green-600 font-medium mb-4">
            <span className="w-5 h-5 bg-green-100 rounded-full flex items-center justify-center text-xs">✓</span>
            Profile generated
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-gray-900">
            Your AI profile is ready{business.name ? `, ${business.name}` : ''}!
          </h1>
        </div>

        {/* llms.txt preview */}
        <div className="mb-8">
          <p className="text-sm font-medium text-gray-500 mb-2">Your generated llms.txt:</p>
          <div className="bg-gray-50 rounded-xl border border-gray-200 p-4 overflow-auto max-h-64">
            <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono leading-relaxed">
              {business.llms_txt_content}
            </pre>
          </div>
        </div>

        {/* Email capture */}
        <div className="bg-white border-2 border-gray-100 rounded-2xl p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-1">
            Where should we send your leads?
          </h2>
          <p className="text-sm text-gray-500 mb-6">
            We&apos;ll notify you by email when we find a relevant business in our network.
          </p>

          <form onSubmit={handleSubmit} className="space-y-4">
            <input
              type="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="your@email.com"
              required
              className="w-full px-4 py-3 text-base border-2 border-gray-200 rounded-xl focus:outline-none focus:border-gray-900 transition-colors"
            />

            <label className="flex items-start gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={agreed}
                onChange={e => setAgreed(e.target.checked)}
                className="mt-1 w-4 h-4 accent-gray-900 flex-shrink-0"
              />
              <span className="text-sm text-gray-600">
                I agree to receive B2B contact opportunities via Meshly
              </span>
            </label>

            {error && <p className="text-sm text-red-500">{error}</p>}

            <button
              type="submit"
              disabled={loading || !email || !agreed}
              className="w-full py-3.5 bg-gray-900 text-white font-semibold rounded-xl hover:bg-gray-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-base"
            >
              {loading ? 'Joining...' : 'Join the network →'}
            </button>

            <p className="text-xs text-center text-gray-400">
              Free forever. Unsubscribe anytime.
            </p>
          </form>
        </div>
      </main>
    </div>
  )
}

export default function SetupPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="w-6 h-6 border-2 border-gray-900 border-t-transparent rounded-full animate-spin" />
      </div>
    }>
      <SetupContent />
    </Suspense>
  )
}
