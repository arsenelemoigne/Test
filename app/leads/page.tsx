import { redirect } from 'next/navigation'
import { createClient } from '@/lib/supabase/server'
import Link from 'next/link'

type LeadRow = {
  id: string
  status: string
  match_score: number
  match_reason: string
  created_at: string
  sender: { name?: string; industry?: string; location?: string } | null
}

export default async function LeadsPage() {
  const supabase = createClient()

  const { data: { user } } = await supabase.auth.getUser()
  if (!user) redirect('/?auth_required=1')

  const { data: business } = await supabase
    .from('businesses')
    .select('id, name')
    .eq('user_id', user.id)
    .single()

  if (!business) redirect('/')

  const { data: leads } = await supabase
    .from('leads')
    .select(`
      id, created_at, match_score, match_reason, status,
      sender:sender_business_id(id, name, industry, location, offering_summary)
    `)
    .eq('recipient_business_id', business.id)
    .order('created_at', { ascending: false })

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <span className="text-xl font-bold tracking-tight">Meshly</span>
          <div className="flex items-center gap-6 text-sm">
            <Link href="/dashboard" className="text-gray-500 hover:text-gray-900">Dashboard</Link>
            <Link href="/profile" className="text-gray-500 hover:text-gray-900">Profile</Link>
            <Link href="/leads" className="font-medium text-gray-900">Leads</Link>
          </div>
        </div>
      </nav>

      <main className="max-w-3xl mx-auto px-6 py-10">
        <h1 className="text-2xl font-bold text-gray-900 mb-8">Your leads</h1>

        {!leads?.length ? (
          <div className="bg-white rounded-2xl border border-gray-200 p-12 text-center">
            <p className="text-gray-400 mb-4">No leads yet.</p>
            <Link
              href="/profile"
              className="text-sm font-medium text-gray-900 underline"
            >
              Complete your profile to get matches →
            </Link>
          </div>
        ) : (
          <div className="space-y-3">
            {(leads as unknown as LeadRow[]).map((lead) => (
              <div
                key={lead.id}
                className="bg-white rounded-2xl border border-gray-200 p-5"
              >
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <p className="font-semibold text-gray-900">{lead.sender?.name}</p>
                    <p className="text-sm text-gray-500">
                      {lead.sender?.industry} · {lead.sender?.location}
                    </p>
                  </div>
                  <span className={`text-xs px-2.5 py-1 rounded-full font-medium ${
                    lead.status === 'interested'
                      ? 'bg-green-100 text-green-700'
                      : lead.status === 'declined'
                      ? 'bg-gray-100 text-gray-500'
                      : 'bg-blue-50 text-blue-600'
                  }`}>
                    {lead.status}
                  </span>
                </div>

                <p className="text-sm text-gray-600 italic mb-3">
                  &quot;{lead.match_reason}&quot;
                </p>

                <div className="flex items-center gap-3 mb-4">
                  <div className="flex-1 bg-gray-100 rounded-full h-1.5">
                    <div
                      className="bg-gray-900 h-1.5 rounded-full"
                      style={{ width: `${Math.round(lead.match_score * 100)}%` }}
                    />
                  </div>
                  <span className="text-sm text-gray-500">
                    {Math.round(lead.match_score * 100)}% match
                  </span>
                  <span className="text-xs text-gray-400">
                    {new Date(lead.created_at).toLocaleDateString()}
                  </span>
                </div>

                {lead.status === 'sent' || lead.status === 'opened' ? (
                  <div className="flex gap-2">
                    <Link
                      href={`/leads/${lead.id}?interested=1`}
                      className="flex-1 text-center py-2.5 bg-gray-900 text-white rounded-xl text-sm font-medium hover:bg-gray-700 transition-colors"
                    >
                      ★ Interested
                    </Link>
                    <Link
                      href={`/api/leads/${lead.id}/declined`}
                      className="flex-1 text-center py-2.5 bg-gray-100 text-gray-600 rounded-xl text-sm hover:bg-gray-200 transition-colors"
                    >
                      ✗ Not relevant
                    </Link>
                  </div>
                ) : lead.status === 'interested' ? (
                  <Link
                    href={`/leads/${lead.id}`}
                    className="block text-center py-2.5 bg-green-50 text-green-700 rounded-xl text-sm font-medium hover:bg-green-100 transition-colors"
                  >
                    View contact details →
                  </Link>
                ) : null}
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  )
}
