import { NextRequest, NextResponse } from 'next/server'
import { createAdminClient } from '@/lib/supabase/admin'
import { sendWelcomeEmail } from '@/lib/emails'

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

  try {
    const result = await sendWelcomeEmail({
      to: email,
      businessName: business.name || 'Your business',
    })
    console.log('Welcome email result:', JSON.stringify(result))
  } catch (err) {
    console.error('Welcome email FAILED:', JSON.stringify(err))
  }

  return NextResponse.json({ success: true })
}
