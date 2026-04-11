import { NextRequest, NextResponse } from 'next/server'
import { createAdminClient } from '@/lib/supabase/admin'
import { sendWelcomeEmail } from '@/lib/emails'
import { createClient as createSupabaseClient } from '@supabase/supabase-js'

export async function POST(request: NextRequest) {
  const body = await request.json()
  const { business_id, email } = body

  if (!business_id || !email) {
    return NextResponse.json(
      { error: 'Missing required fields.' },
      { status: 400 }
    )
  }

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  if (!emailRegex.test(email)) {
    return NextResponse.json({ error: 'Invalid email address.' }, { status: 400 })
  }

  const supabase = createAdminClient()

  // Check if welcome email was already sent for this business
  const { data: existing } = await supabase
    .from('businesses')
    .select('email')
    .eq('id', business_id)
    .single()

  const alreadyActivated = !!(existing?.email)

  const { data: business, error } = await supabase
    .from('businesses')
    .update({ email })
    .eq('id', business_id)
    .select('id, name, slug, email')
    .single()

  if (error || !business) {
    console.error('Business lookup failed:', error)
    return NextResponse.json(
      { error: 'Business not found.' },
      { status: 404 }
    )
  }

  // Only send welcome email on first activation
  if (!alreadyActivated) {
    // Generate magic link so "Complete my profile" logs the user in automatically
    let profileUrl: string | undefined
    try {
      const authClient = createSupabaseClient(
        process.env.NEXT_PUBLIC_SUPABASE_URL!,
        process.env.SUPABASE_SERVICE_ROLE_KEY!,
        { auth: { autoRefreshToken: false, persistSession: false } }
      )

      const { data: linkData } = await authClient.auth.admin.generateLink({
        type: 'magiclink',
        email,
        options: {
          redirectTo: `${process.env.NEXT_PUBLIC_APP_URL}/profile`,
        },
      })

      profileUrl = linkData?.properties?.action_link ?? undefined
    } catch (err) {
      console.error('Magic link generation failed:', err)
    }

    try {
      const result = await sendWelcomeEmail({
        to: email,
        businessName: business.name || 'Your business',
        businessId: business.id,
        profileUrl,
      })
      console.log('Welcome email result:', JSON.stringify(result))
    } catch (err) {
      console.error('Welcome email FAILED:', JSON.stringify(err))
    }
  }

  return NextResponse.json({ success: true })
}
