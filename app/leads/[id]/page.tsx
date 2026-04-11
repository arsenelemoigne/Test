import { redirect } from 'next/navigation'
import { createClient } from '@/lib/supabase/server'
import { createAdminClient } from '@/lib/supabase/admin'
import Link from 'next/link'

export default async function LeadDetailPage({
  params,
  searchParams,
}: {
  params: { id: string }
  searchParams: { interested?: string }
}) {
  const supabase = createClient()

  const { data: { user } } = await supabase.auth.getUser()
  if (!user) redirect('/?auth_required=1')

  const { data: userBusiness } = await supabase
    .from('businesses')
    .select('id')
    .eq('user_id', user.id)
    .single()

  if (!userBusiness) redirect('/')

  // Use admin client to update status if marking as interested
  if (searchParams.interested === '1') {
    const adminClient = createAdminClient()
    await adminClient
      .from('leads')
      .update({ status: 'interested', responded_at: new Date().toISOString() })
      .eq('id', params.id)
      .eq('recipient_business_id', userBusiness.id)
  }

  const { data: lead } = await supabase
    .from('leads')
    .select(`
      *,
      sender:sender_business_id(*),
      recipient:recipient_business_id(id, name)
    `)
    .eq('id', params.id)
    .eq('recipient_business_id', userBusiness.id)
    .single()

  if (!lead) redirect('/leads')

  const sender = lead.sender as { id?: string; name?: string; industry?: string; location?: string; offering_summary?: string; description?: string; services?: string[]; url?: string; email?: string }
  const showContact = lead.status === 'interested'

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <span className="text-xl font-bold tracking-tight">Meshly</span>
          <div className="flex items-center gap-6 text-sm">
            <Link href="/dashboard" className="text-gray-500 hover:text-gray-900">Dashboard</Link>
            <Link href="/leads" className="text-gray-500 hover:text-gray-900">← Back to leads</Link>
          </div>
        </div>
      </nav>

      <main className="max-w-2xl mx-auto px-6 py-10">
        <div className="bg-white rounded-2xl border border-gray-200 p-8">
          {showContact && (
            <div className="bg-green-50 border border-green-200 rounded-xl p-4 mb-6">
              <p className="text-sm font-medium text-green-800">
                ✓ You&apos;re interested — contact details are now visible.
              </p>
            </div>
          )}

          <div className="mb-6">
            <h1 className="text-2xl font-bold text-gray-900">{sender?.name}</h1>
            <p className="text-gray-500 mt-1">
              {sender?.industry} · {sender?.location}
            </p>
          </div>

          <div className="space-y-4 mb-6">
            <div>
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">What they do</p>
              <p className="text-gray-700">{sender?.offering_summary}</p>
            </div>

            {sender?.description && (
              <div>
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">About</p>
                <p className="text-gray-700 text-sm">{sender?.description}</p>
              </div>
            )}

            {(sender?.services?.length ?? 0) > 0 && (
              <div>
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">Services</p>
                <div className="flex flex-wrap gap-2">
                  {(sender.services ?? []).map((s: string) => (
                    <span key={s} className="px-3 py-1 bg-gray-100 rounded-full text-sm text-gray-700">
                      {s}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="border-t border-gray-100 pt-5 mb-6">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Match reason</p>
            <p className="text-gray-700 italic">&quot;{lead.match_reason}&quot;</p>
            <div className="flex items-center gap-3 mt-3">
              <div className="flex-1 bg-gray-100 rounded-full h-2">
                <div
                  className="bg-gray-900 h-2 rounded-full"
                  style={{ width: `${Math.round(lead.match_score * 100)}%` }}
                />
              </div>
              <span className="text-sm text-gray-600 font-medium">
                {Math.round(lead.match_score * 100)}% match
              </span>
            </div>
          </div>

          {/* Contact info — only visible after "Interested" */}
          {showContact ? (
            <div className="bg-gray-50 rounded-xl p-5 mb-6">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-3">Contact</p>
              <div className="space-y-2">
                {sender?.url && (
                  <div>
                    <p className="text-xs text-gray-400">Website</p>
                    <a
                      href={sender.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-blue-600 hover:underline"
                    >
                      {sender.url}
                    </a>
                  </div>
                )}
                {sender?.email && (
                  <div>
                    <p className="text-xs text-gray-400">Email</p>
                    <a
                      href={`mailto:${sender.email}`}
                      className="text-sm text-blue-600 hover:underline"
                    >
                      {sender.email}
                    </a>
                  </div>
                )}
              </div>

              {sender?.email && (
                <a
                  href={`mailto:${sender.email}?subject=Introduction via Meshly&body=Hi ${sender.name},%0D%0A%0D%0AI found your profile on Meshly and would love to connect.%0D%0A%0D%0A`}
                  className="mt-4 block text-center py-3 bg-gray-900 text-white rounded-xl text-sm font-medium hover:bg-gray-700 transition-colors"
                >
                  Reply by email →
                </a>
              )}
            </div>
          ) : (
            <div className="bg-gray-50 rounded-xl p-5 mb-6 text-center">
              <p className="text-sm text-gray-500 mb-3">
                Mark as interested to reveal contact details.
              </p>
              <Link
                href={`/leads/${lead.id}?interested=1`}
                className="inline-block px-6 py-2.5 bg-gray-900 text-white rounded-xl text-sm font-medium hover:bg-gray-700 transition-colors"
              >
                ★ I&apos;m interested
              </Link>
            </div>
          )}

          <div className="flex gap-2">
            <Link href="/leads" className="flex-1 text-center py-2.5 border border-gray-200 rounded-xl text-sm text-gray-600 hover:bg-gray-50 transition-colors">
              ← Back to leads
            </Link>
            {lead.status !== 'declined' && (
              <Link
                href={`/api/leads/${lead.id}/declined`}
                className="px-4 py-2.5 border border-gray-200 rounded-xl text-sm text-gray-400 hover:bg-gray-50 transition-colors"
              >
                ✗ Not relevant
              </Link>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
