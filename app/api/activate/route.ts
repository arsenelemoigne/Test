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

  // Basic email validation
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  if (!emailRegex.test(email)) {
    return NextResponse.json({ error: 'Invalid email address.' }, { status: 400 })
  }

  const supabase = createAdminClient()

  // Update business with email
  const { data: business, error } = await supabase
    .from('businesses')
    .update({ email })
    .eq('id', business_id)
    .select('id, name, slug, email')
    .single()

  if (error || !business) {
    return NextResponse.json(
      { error: 'Business not found.' },
      { status: 404 }
    )
  }

  // Send welcome email via Resend
  try {
    await sendWelcomeEmail({
      to: email,
      businessName: business.name || 'Your business',
      businessId: business.id,
    })
  } catch (err) {
    console.error('Welcome email failed:', err)
    // Don't fail the whole request if email fails
  }

  // Send magic link for platform login (Supabase Auth)
  try {
    const authClient = createSupabaseClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!,
      { auth: { autoRefreshToken: false, persistSession: false } }
    )

    await authClient.auth.admin.generateLink({
      type: 'magiclink',
      email,
      options: {
        redirectTo: `${process.env.NEXT_PUBLIC_APP_URL}/dashboard?business_id=${business_id}`,
      },
    })
  } catch (err) {
    console.error('Magic link failed:', err)
    // Non-blocking
  }

  return NextResponse.json({ success: true })
}
