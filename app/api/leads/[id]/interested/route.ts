import { NextRequest, NextResponse } from 'next/server'
import { createAdminClient } from '@/lib/supabase/admin'

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const { id } = params
  const supabase = createAdminClient()

  const { data: lead, error } = await supabase
    .from('leads')
    .update({
      status: 'interested',
      responded_at: new Date().toISOString(),
    })
    .eq('id', id)
    .select('id, recipient_business_id')
    .single()

  if (error || !lead) {
    return new NextResponse('Lead not found', { status: 404 })
  }

  const appUrl = process.env.NEXT_PUBLIC_APP_URL || 'https://meshly.com'
  return NextResponse.redirect(`${appUrl}/leads/${id}?interested=1`)
}
