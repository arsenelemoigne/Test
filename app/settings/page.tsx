import { redirect } from 'next/navigation'
import { createClient } from '@/lib/supabase/server'
import Link from 'next/link'

export default async function SettingsPage() {
  const supabase = createClient()

  const { data: { user } } = await supabase.auth.getUser()
  if (!user) redirect('/?auth_required=1')

  const { data: business } = await supabase
    .from('businesses')
    .select('id, name, email, is_active, url, slug')
    .eq('user_id', user.id)
    .single()

  if (!business) redirect('/')

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <span className="text-xl font-bold tracking-tight">Meshly</span>
          <div className="flex items-center gap-6 text-sm">
            <Link href="/dashboard" className="text-gray-500 hover:text-gray-900">Dashboard</Link>
            <Link href="/profile" className="text-gray-500 hover:text-gray-900">Profile</Link>
            <Link href="/leads" className="text-gray-500 hover:text-gray-900">Leads</Link>
            <Link href="/settings" className="font-medium text-gray-900">Settings</Link>
          </div>
        </div>
      </nav>

      <main className="max-w-2xl mx-auto px-6 py-10">
        <h1 className="text-2xl font-bold text-gray-900 mb-8">Settings</h1>

        <div className="space-y-4">
          {/* Account */}
          <div className="bg-white rounded-2xl border border-gray-200 p-6">
            <h2 className="font-semibold text-gray-900 mb-4">Account</h2>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-500">Email</span>
                <span className="text-gray-900">{business.email}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Business</span>
                <span className="text-gray-900">{business.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Website</span>
                <a href={business.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline truncate max-w-[200px]">
                  {business.url}
                </a>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Your llms.txt</span>
                <a
                  href={`/b/${business.slug}/llms.txt`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline"
                >
                  /b/{business.slug}/llms.txt
                </a>
              </div>
            </div>
          </div>

          {/* Email preferences */}
          <div className="bg-white rounded-2xl border border-gray-200 p-6">
            <h2 className="font-semibold text-gray-900 mb-2">Email preferences</h2>
            <p className="text-sm text-gray-500 mb-4">
              You&apos;re currently receiving lead notifications at {business.email}.
            </p>
            <p className="text-xs text-gray-400">
              To unsubscribe, contact us or use the unsubscribe link in any email.
            </p>
          </div>

          {/* Network status */}
          <div className="bg-white rounded-2xl border border-gray-200 p-6">
            <h2 className="font-semibold text-gray-900 mb-2">Network status</h2>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${business.is_active ? 'bg-green-500' : 'bg-gray-300'}`} />
              <span className="text-sm text-gray-600">
                {business.is_active ? 'Active — you are visible in the network' : 'Inactive'}
              </span>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
