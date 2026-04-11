import { NextRequest, NextResponse } from 'next/server'
import { createAdminClient } from '@/lib/supabase/admin'

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const { id } = params
  const supabase = createAdminClient()

  await supabase
    .from('leads')
    .update({
      status: 'declined',
      responded_at: new Date().toISOString(),
    })
    .eq('id', id)

  const appUrl = process.env.NEXT_PUBLIC_APP_URL || 'https://meshly.com'
  return NextResponse.redirect(`${appUrl}/thanks-feedback`)
}
