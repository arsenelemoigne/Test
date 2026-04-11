import { NextRequest, NextResponse } from 'next/server'
import { runMatchingEngine } from '@/scripts/matching-engine'

export async function GET(request: NextRequest) {
  const authHeader = request.headers.get('authorization')
  const expected = `Bearer ${process.env.CRON_SECRET}`

  if (authHeader !== expected) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  try {
    await runMatchingEngine()
    return NextResponse.json({ success: true, ran_at: new Date().toISOString() })
  } catch (err) {
    console.error('Matching engine failed:', err)
    return NextResponse.json({ error: 'Matching engine failed' }, { status: 500 })
  }
}
