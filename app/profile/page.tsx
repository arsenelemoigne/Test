'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'
import { calculateProfileCompleteness } from '@/lib/profile-completeness'
import Link from 'next/link'

const INDUSTRIES = [
  'E-commerce', 'Hospitality', 'Construction', 'Health', 'Education',
  'Food & Beverage', 'Tech/SaaS', 'Finance', 'Creative/Agency', 'Retail',
  'Manufacturing', 'Real estate', 'Other',
]

const NEEDS_OPTIONS = [
  'New clients / customers',
  'Product or material suppliers',
  'Software or tools',
  'Freelancers or subcontractors',
  'Strategic partners or resellers',
  'Investors or funding',
  'Nothing specific right now',
]

const LANGUAGES = ['FR', 'EN', 'ES', 'DE', 'IT']

type Section = 1 | 2 | 3 | 4

export default function ProfilePage() {
  const router = useRouter()
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [business, setBusiness] = useState<{ id: string } | null>(null)
  const [section, setSection] = useState<Section>(1)

  // Form state
  const [name, setName] = useState('')
  const [offeringSummary, setOfferingSummary] = useState('')
  const [services, setServices] = useState<string[]>([])
  const [serviceInput, setServiceInput] = useState('')
  const [location, setLocation] = useState('')
  const [languages, setLanguages] = useState<string[]>([])
  const [idealCustomer, setIdealCustomer] = useState('')
  const [customerIndustries, setCustomerIndustries] = useState<string[]>([])
  const [customerSize, setCustomerSize] = useState('')
  const [geoScope, setGeoScope] = useState('')
  const [needs, setNeeds] = useState<string[]>([])
  const [needsDetail, setNeedsDetail] = useState('')
  const [keywords, setKeywords] = useState<string[]>([])
  const [keywordInput, setKeywordInput] = useState('')

  useEffect(() => {
    const supabase = createClient()
    supabase.auth.getUser().then(({ data: { user } }) => {
      if (!user) {
        router.push('/')
        return
      }
      supabase
        .from('businesses')
        .select('*')
        .eq('user_id', user.id)
        .single()
        .then(({ data }) => {
          if (!data) { router.push('/'); return }
          setBusiness(data)
          // Pre-fill from scraped data
          setName(data.name || '')
          setOfferingSummary(data.offering_summary || '')
          setServices(data.services || [])
          setLocation(data.location || '')
          setLanguages(data.languages || [])
          setIdealCustomer(data.ideal_customer_profile || '')
          setCustomerIndustries(data.customer_industries || [])
          setCustomerSize(data.customer_size || '')
          setGeoScope(data.geographic_scope || '')
          setNeeds(data.needs || [])
          setKeywords(data.keywords || [])
          setLoading(false)
        })
    })
  }, [router])

  async function handleSave() {
    if (!business) return
    setSaving(true)

    const updates = {
      name,
      offering_summary: offeringSummary,
      services,
      location,
      languages,
      ideal_customer_profile: idealCustomer,
      customer_industries: customerIndustries,
      customer_size: customerSize,
      geographic_scope: geoScope,
      needs,
      keywords,
    }

    const completeness = calculateProfileCompleteness({
      ...business,
      ...updates,
    })

    const supabase = createClient()
    await supabase
      .from('businesses')
      .update({ ...updates, profile_completeness: completeness })
      .eq('id', business.id)

    setSaving(false)

    if (section < 4) {
      setSection((section + 1) as Section)
    } else {
      router.push('/dashboard')
    }
  }

  function addTag(value: string, setter: (fn: (prev: string[]) => string[]) => void) {
    const trimmed = value.trim()
    if (trimmed) {
      setter(prev => prev.includes(trimmed) ? prev : [...prev, trimmed])
    }
  }

  function removeTag(value: string, setter: (fn: (prev: string[]) => string[]) => void) {
    setter(prev => prev.filter(t => t !== value))
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="w-6 h-6 border-2 border-gray-900 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  const sectionTitles = ['What you offer', 'Who you sell to', "What you're looking for", 'Keywords']

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <span className="text-xl font-bold tracking-tight">Meshly</span>
          <div className="flex items-center gap-6 text-sm">
            <Link href="/dashboard" className="text-gray-500 hover:text-gray-900">Dashboard</Link>
            <Link href="/profile" className="font-medium text-gray-900">Profile</Link>
            <Link href="/leads" className="text-gray-500 hover:text-gray-900">Leads</Link>
          </div>
        </div>
      </nav>

      <main className="max-w-2xl mx-auto px-6 py-10">
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-gray-900 mb-1">Edit your profile</h1>
          <p className="text-gray-500 text-sm">
            A complete profile gets better matches.
          </p>
        </div>

        {/* Section tabs */}
        <div className="flex gap-1 mb-8 bg-gray-100 rounded-xl p-1">
          {([1, 2, 3, 4] as Section[]).map(s => (
            <button
              key={s}
              onClick={() => setSection(s)}
              className={`flex-1 py-2 text-xs font-medium rounded-lg transition-colors ${
                section === s
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              {sectionTitles[s - 1]}
            </button>
          ))}
        </div>

        <div className="bg-white rounded-2xl border border-gray-200 p-6">
          {/* Section 1 — What you offer */}
          {section === 1 && (
            <div className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1.5">
                  Business name
                </label>
                <input
                  value={name}
                  onChange={e => setName(e.target.value)}
                  className="w-full px-4 py-2.5 border border-gray-200 rounded-xl focus:outline-none focus:border-gray-900 text-sm"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1.5">
                  One-line pitch
                  <span className="text-gray-400 font-normal ml-1">(140 chars max)</span>
                </label>
                <input
                  value={offeringSummary}
                  onChange={e => setOfferingSummary(e.target.value.slice(0, 140))}
                  placeholder="We offer _____ to _____"
                  className="w-full px-4 py-2.5 border border-gray-200 rounded-xl focus:outline-none focus:border-gray-900 text-sm"
                />
                <p className="text-xs text-gray-400 mt-1 text-right">
                  {offeringSummary.length}/140
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1.5">Services</label>
                <div className="flex flex-wrap gap-2 mb-2">
                  {services.map(s => (
                    <span key={s} className="inline-flex items-center gap-1 px-3 py-1 bg-gray-100 rounded-full text-sm">
                      {s}
                      <button onClick={() => removeTag(s, setServices)} className="text-gray-400 hover:text-gray-900 text-xs ml-0.5">×</button>
                    </span>
                  ))}
                </div>
                <div className="flex gap-2">
                  <input
                    value={serviceInput}
                    onChange={e => setServiceInput(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') {
                        e.preventDefault()
                        addTag(serviceInput, setServices)
                        setServiceInput('')
                      }
                    }}
                    placeholder="Add a service, press Enter"
                    className="flex-1 px-4 py-2.5 border border-gray-200 rounded-xl focus:outline-none focus:border-gray-900 text-sm"
                  />
                  <button
                    onClick={() => { addTag(serviceInput, setServices); setServiceInput('') }}
                    className="px-4 py-2.5 bg-gray-100 rounded-xl text-sm font-medium hover:bg-gray-200 transition-colors"
                  >
                    Add
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1.5">Location</label>
                <input
                  value={location}
                  onChange={e => setLocation(e.target.value)}
                  placeholder="City, Country"
                  className="w-full px-4 py-2.5 border border-gray-200 rounded-xl focus:outline-none focus:border-gray-900 text-sm"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Languages</label>
                <div className="flex flex-wrap gap-2">
                  {LANGUAGES.map(lang => (
                    <label key={lang} className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={languages.includes(lang)}
                        onChange={e => {
                          setLanguages(prev =>
                            e.target.checked ? [...prev, lang] : prev.filter(l => l !== lang)
                          )
                        }}
                        className="accent-gray-900"
                      />
                      <span className="text-sm">{lang}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Section 2 — Who you sell to */}
          {section === 2 && (
            <div className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1.5">
                  Describe your ideal client
                  <span className="text-gray-400 font-normal ml-1">(200 chars max)</span>
                </label>
                <textarea
                  value={idealCustomer}
                  onChange={e => setIdealCustomer(e.target.value.slice(0, 200))}
                  placeholder="Marketing managers at mid-size retail brands in France"
                  rows={3}
                  className="w-full px-4 py-2.5 border border-gray-200 rounded-xl focus:outline-none focus:border-gray-900 text-sm resize-none"
                />
                <p className="text-xs text-gray-400 mt-1 text-right">{idealCustomer.length}/200</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Industries you target</label>
                <div className="grid grid-cols-2 gap-2">
                  {INDUSTRIES.map(industry => (
                    <label key={industry} className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={customerIndustries.includes(industry)}
                        onChange={e => {
                          setCustomerIndustries(prev =>
                            e.target.checked ? [...prev, industry] : prev.filter(i => i !== industry)
                          )
                        }}
                        className="accent-gray-900"
                      />
                      <span className="text-sm">{industry}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Client size</label>
                <div className="flex flex-wrap gap-2">
                  {['Solo/Freelance', 'Small (2-20)', 'Medium (20-200)', 'Large (200+)', 'Any'].map(size => (
                    <label key={size} className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        name="customerSize"
                        checked={customerSize === size}
                        onChange={() => setCustomerSize(size)}
                        className="accent-gray-900"
                      />
                      <span className="text-sm">{size}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Geographic scope</label>
                <div className="flex flex-wrap gap-4">
                  {[
                    { value: 'local', label: 'Local (my city)' },
                    { value: 'national', label: 'National' },
                    { value: 'europe', label: 'Europe' },
                    { value: 'international', label: 'International' },
                  ].map(opt => (
                    <label key={opt.value} className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        name="geoScope"
                        checked={geoScope === opt.value}
                        onChange={() => setGeoScope(opt.value)}
                        className="accent-gray-900"
                      />
                      <span className="text-sm">{opt.label}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Section 3 — What you're looking for */}
          {section === 3 && (
            <div className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Are you currently looking for anything?
                </label>
                <div className="space-y-2">
                  {NEEDS_OPTIONS.map(need => (
                    <label key={need} className="flex items-center gap-3 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={needs.includes(need)}
                        onChange={e => {
                          setNeeds(prev =>
                            e.target.checked ? [...prev, need] : prev.filter(n => n !== need)
                          )
                        }}
                        className="accent-gray-900"
                      />
                      <span className="text-sm">{need}</span>
                    </label>
                  ))}
                </div>
              </div>

              {needs.length > 0 && needs[0] !== 'Nothing specific right now' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1.5">
                    Tell us more
                    <span className="text-gray-400 font-normal ml-1">(optional, 300 chars max)</span>
                  </label>
                  <textarea
                    value={needsDetail}
                    onChange={e => setNeedsDetail(e.target.value.slice(0, 300))}
                    placeholder="More details about what you're looking for..."
                    rows={3}
                    className="w-full px-4 py-2.5 border border-gray-200 rounded-xl focus:outline-none focus:border-gray-900 text-sm resize-none"
                  />
                  <p className="text-xs text-gray-400 mt-1 text-right">{needsDetail.length}/300</p>
                </div>
              )}
            </div>
          )}

          {/* Section 4 — Keywords */}
          {section === 4 && (
            <div className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1.5">
                  Add keywords that describe your business
                </label>
                <p className="text-xs text-gray-400 mb-3">
                  Examples: &quot;vegan&quot;, &quot;B2B SaaS&quot;, &quot;wholesale&quot;, &quot;Paris 10e&quot;, &quot;luxury&quot;
                </p>
                <div className="flex flex-wrap gap-2 mb-3">
                  {keywords.map(kw => (
                    <span key={kw} className="inline-flex items-center gap-1 px-3 py-1 bg-gray-100 rounded-full text-sm">
                      {kw}
                      <button onClick={() => removeTag(kw, setKeywords)} className="text-gray-400 hover:text-gray-900 text-xs ml-0.5">×</button>
                    </span>
                  ))}
                </div>
                <div className="flex gap-2">
                  <input
                    value={keywordInput}
                    onChange={e => setKeywordInput(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') {
                        e.preventDefault()
                        addTag(keywordInput, setKeywords)
                        setKeywordInput('')
                      }
                    }}
                    placeholder="Add a keyword, press Enter"
                    className="flex-1 px-4 py-2.5 border border-gray-200 rounded-xl focus:outline-none focus:border-gray-900 text-sm"
                  />
                  <button
                    onClick={() => { addTag(keywordInput, setKeywords); setKeywordInput('') }}
                    className="px-4 py-2.5 bg-gray-100 rounded-xl text-sm font-medium hover:bg-gray-200 transition-colors"
                  >
                    Add
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center justify-between mt-6 pt-6 border-t border-gray-100">
            {section > 1 ? (
              <button
                onClick={() => setSection((section - 1) as Section)}
                className="text-sm text-gray-500 hover:text-gray-900"
              >
                ← Back
              </button>
            ) : <div />}

            <button
              onClick={handleSave}
              disabled={saving}
              className="px-6 py-2.5 bg-gray-900 text-white font-medium rounded-xl hover:bg-gray-700 transition-colors disabled:opacity-50 text-sm"
            >
              {saving ? 'Saving...' : section < 4 ? 'Save & continue →' : 'Save profile ✓'}
            </button>
          </div>
        </div>
      </main>
    </div>
  )
}
