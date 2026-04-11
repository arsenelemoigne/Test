import { redirect } from 'next/navigation'
import { createClient } from '@/lib/supabase/server'
import Link from 'next/link'

type RecentLead = {
  id: string
  status: string
  match_score: number
  match_reason: string
  sender: { name?: string; industry?: string; location?: string } | null
}

export default async function DashboardPage() {
  const supabase = createClient()

  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    redirect('/?auth_required=1')
  }

  // Get user's business
  const { data: business } = await supabase
    .from('businesses')
    .select('id, name, profile_completeness')
    .eq('user_id', user.id)
    .single()

  if (!business) {
    redirect('/')
  }

  // Get stats in parallel
  const [leadsResult, llmVisitsResult, recentLeadsResult, visitBreakdownResult] =
    await Promise.all([
      supabase
        .from('leads')
        .select('id', { count: 'exact' })
        .eq('recipient_business_id', business.id),
      supabase
        .from('llm_visits')
        .select('id', { count: 'exact' })
        .eq('business_id', business.id)
        .not('bot_name', 'is', null),
      supabase
        .from('leads')
        .select(`
          id, created_at, match_score, match_reason, status,
          sender:sender_business_id(name, industry, location)
        `)
        .eq('recipient_business_id', business.id)
        .order('created_at', { ascending: false })
        .limit(3),
      supabase
        .from('llm_visits')
        .select('bot_name, visited_at')
        .eq('business_id', business.id)
        .not('bot_name', 'is', null)
        .order('visited_at', { ascending: false })
        .limit(200),
    ])

  const totalLeads = leadsResult.count || 0
  const totalAiVisits = llmVisitsResult.count || 0
  const recentLeads = recentLeadsResult.data || []
  const visitData = visitBreakdownResult.data || []

  // Bot visit breakdown
  const botCounts: Record<string, number> = {}
  for (const v of visitData) {
    if (v.bot_name) {
      botCounts[v.bot_name] = (botCounts[v.bot_name] || 0) + 1
    }
  }
  const botEntries = Object.entries(botCounts).sort((a, b) => b[1] - a[1])
  const maxBotVisits = botEntries[0]?.[1] || 1

  const lastVisit = visitData[0]
  const lastVisitTime = lastVisit
    ? formatRelativeTime(new Date(lastVisit.visited_at))
    : null

  // Visits this week
  const oneWeekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
  const visitsThisWeek = visitData.filter(
    v => new Date(v.visited_at) > oneWeekAgo
  ).length

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Nav */}
      <nav className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <span className="text-xl font-bold tracking-tight">Meshly</span>
          <div className="flex items-center gap-6 text-sm">
            <Link href="/dashboard" className="font-medium text-gray-900">
              Dashboard
            </Link>
            <Link href="/profile" className="text-gray-500 hover:text-gray-900">
              Profile
            </Link>
            <Link href="/leads" className="text-gray-500 hover:text-gray-900">
              Leads
            </Link>
            <Link href="/settings" className="text-gray-500 hover:text-gray-900">
              Settings
            </Link>
          </div>
        </div>
      </nav>

      <main className="max-w-5xl mx-auto px-6 py-10">
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-gray-900">
            Welcome back, {business.name}
          </h1>
        </div>

        {/* Profile completeness banner */}
        {business.profile_completeness < 80 && (
          <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 mb-8 flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-amber-800">
                Your profile is {business.profile_completeness}% complete.
              </p>
              <p className="text-sm text-amber-700">
                Complete it to get better matches.
              </p>
            </div>
            <Link
              href="/profile"
              className="text-sm font-medium text-amber-900 bg-amber-100 px-4 py-2 rounded-lg hover:bg-amber-200 transition-colors whitespace-nowrap"
            >
              Complete profile →
            </Link>
          </div>
        )}

        {/* Stats */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
          <StatCard
            label="Leads received"
            value={totalLeads}
            icon="✉"
          />
          <StatCard
            label="AI visits"
            value={totalAiVisits}
            icon="🤖"
          />
          <StatCard
            label="Profile strength"
            value={`${business.profile_completeness}%`}
            icon="★"
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Recent leads */}
          <div className="bg-white rounded-2xl border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-semibold text-gray-900">Recent leads</h2>
              <Link href="/leads" className="text-sm text-gray-500 hover:text-gray-900">
                View all
              </Link>
            </div>

            {recentLeads.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-gray-400 text-sm">No leads yet.</p>
                <Link href="/profile" className="text-sm text-gray-900 font-medium mt-2 inline-block">
                  Complete your profile to get matches →
                </Link>
              </div>
            ) : (
              <div className="space-y-3">
                {(recentLeads as unknown as RecentLead[]).map((lead) => (
                  <Link
                    key={lead.id}
                    href={`/leads/${lead.id}`}
                    className="block p-3 rounded-xl border border-gray-100 hover:border-gray-300 transition-colors"
                  >
                    <div className="flex items-start justify-between">
                      <div>
                        <p className="font-medium text-gray-900 text-sm">
                          {lead.sender?.name}
                        </p>
                        <p className="text-xs text-gray-500">
                          {lead.sender?.industry} · {lead.sender?.location}
                        </p>
                      </div>
                      <span className={`text-xs px-2 py-0.5 rounded-full ${
                        lead.status === 'interested'
                          ? 'bg-green-100 text-green-700'
                          : lead.status === 'declined'
                          ? 'bg-gray-100 text-gray-500'
                          : 'bg-blue-50 text-blue-600'
                      }`}>
                        {lead.status}
                      </span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1 line-clamp-1">
                      {lead.match_reason}
                    </p>
                    <div className="flex items-center gap-2 mt-2">
                      <div className="flex-1 bg-gray-100 rounded-full h-1">
                        <div
                          className="bg-gray-900 h-1 rounded-full"
                          style={{ width: `${Math.round(lead.match_score * 100)}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-400">
                        {Math.round(lead.match_score * 100)}%
                      </span>
                    </div>
                  </Link>
                ))}
              </div>
            )}
          </div>

          {/* AI visits breakdown */}
          <div className="bg-white rounded-2xl border border-gray-200 p-6">
            <h2 className="font-semibold text-gray-900 mb-1">AI Visits</h2>
            <p className="text-sm text-gray-500 mb-4">
              {visitsThisWeek} this week · {totalAiVisits} all time
            </p>

            {botEntries.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-gray-400 text-sm">No AI visits yet.</p>
                <p className="text-xs text-gray-400 mt-1">
                  Visits will appear once AI assistants discover your profile.
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {botEntries.map(([bot, count]) => (
                  <div key={bot} className="flex items-center gap-3">
                    <span className="text-sm text-gray-600 w-36 truncate">{bot}</span>
                    <div className="flex-1 bg-gray-100 rounded-full h-2">
                      <div
                        className="bg-gray-900 h-2 rounded-full"
                        style={{ width: `${(count / maxBotVisits) * 100}%` }}
                      />
                    </div>
                    <span className="text-sm text-gray-500 w-8 text-right">{count}</span>
                  </div>
                ))}
              </div>
            )}

            {lastVisit && (
              <p className="text-xs text-gray-400 mt-4">
                Last visited: {lastVisitTime} by {lastVisit.bot_name}
              </p>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

function StatCard({ label, value, icon }: { label: string; value: number | string; icon: string }) {
  return (
    <div className="bg-white rounded-2xl border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm text-gray-500">{label}</span>
        <span className="text-lg">{icon}</span>
      </div>
      <p className="text-3xl font-bold text-gray-900">{value}</p>
    </div>
  )
}

function formatRelativeTime(date: Date): string {
  const diffMs = Date.now() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMins / 60)
  const diffDays = Math.floor(diffHours / 24)

  if (diffMins < 1) return 'just now'
  if (diffMins < 60) return `${diffMins} minutes ago`
  if (diffHours < 24) return `${diffHours} hours ago`
  return `${diffDays} days ago`
}
