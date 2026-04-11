import OpenAI from 'openai'
import { createAdminClient } from '@/lib/supabase/admin'
import { areIndustriesCompatible } from '@/lib/industry-affinity'
import { sendLeadEmail } from '@/lib/emails'
import { Business } from '@/lib/types'

function getOpenAI() {
  return new OpenAI({ apiKey: process.env.OPENAI_API_KEY! })
}

interface MatchScore {
  score: number
  reason: string
  match_type: string
  send: boolean
}

async function scoreMatch(
  business: Business,
  candidate: Business
): Promise<MatchScore> {
  const prompt = `You are a B2B matchmaker. Evaluate whether Business B would be a relevant contact for Business A to receive. Score from 0.0 to 1.0.

Business A (the recipient of this potential lead):
- Name: ${business.name}
- Industry: ${business.industry}
- Services: ${business.services?.join(', ')}
- Location: ${business.location}
- Ideal customer: ${business.ideal_customer_profile || 'not specified'}
- Target industries: ${business.customer_industries?.join(', ') || 'not specified'}
- Currently looking for: ${business.needs?.join(', ') || 'inferred: ' + business.inferred_needs?.join(', ')}

Business B (the potential lead to send to Business A):
- Name: ${candidate.name}
- Industry: ${candidate.industry}
- Services: ${candidate.services?.join(', ')}
- Location: ${candidate.location}
- What they offer: ${candidate.offering_summary}

Score this match. Consider:
- Does B offer something A needs (explicit or inferred)?
- Does A's profile match B's ideal client?
- Are they geographically compatible?
- Is there a logical commercial relationship between these two industries?

Return ONLY valid JSON:
{
  "score": 0.0 to 1.0,
  "reason": "One sentence explaining why this is a match",
  "match_type": "A needs B's services | B needs A's services | mutual interest",
  "send": true or false
}

Only return send: true if score >= 0.65 AND there is a clear, specific reason.
Do not match businesses that simply exist in the same city with no logical connection.`

  const openai = getOpenAI()
  const response = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    max_tokens: 200,
    response_format: { type: 'json_object' },
    messages: [{ role: 'user', content: prompt }],
  })

  const text = response.choices[0]?.message?.content || '{}'
  return JSON.parse(text) as MatchScore
}

async function checkAlreadyMatched(
  businessAId: string,
  businessBId: string
): Promise<boolean> {
  const supabase = createAdminClient()
  const { data } = await supabase
    .from('match_queue')
    .select('id')
    .or(
      `and(business_a_id.eq.${businessAId},business_b_id.eq.${businessBId}),and(business_a_id.eq.${businessBId},business_b_id.eq.${businessAId})`
    )
    .limit(1)

  return (data?.length ?? 0) > 0
}

async function processMatchQueue() {
  const supabase = createAdminClient()

  const { data: pending } = await supabase
    .from('match_queue')
    .select(
      `
      *,
      business_a:business_a_id(id, name, industry, location, offering_summary, email),
      business_b:business_b_id(id, name, industry, location, offering_summary, email)
    `
    )
    .eq('processed', false)
    .eq('sent', false)
    .order('score', { ascending: false })
    .limit(50)

  if (!pending?.length) return

  for (const match of pending) {
    const businessA = match.business_a as Business & { email: string }
    const businessB = match.business_b as Business

    if (!businessA.email) {
      // Can't send email, mark as processed
      await supabase
        .from('match_queue')
        .update({ processed: true })
        .eq('id', match.id)
      continue
    }

    // Insert lead record
    const { data: lead } = await supabase
      .from('leads')
      .insert({
        recipient_business_id: businessA.id,
        sender_business_id: businessB.id,
        match_score: match.score,
        match_reason: match.match_reason,
        status: 'sent',
        email_sent_at: new Date().toISOString(),
      })
      .select('id')
      .single()

    if (!lead) continue

    try {
      await sendLeadEmail({
        to: businessA.email,
        recipientBusinessName: businessA.name || 'there',
        senderBusiness: {
          name: businessB.name || 'Unknown',
          industry: businessB.industry || 'Unknown',
          location: businessB.location || 'Unknown',
          offering_summary: businessB.offering_summary || '',
        },
        matchReason: match.match_reason || 'Potential business synergy',
        matchScore: match.score,
        leadId: lead.id,
      })

      await supabase
        .from('match_queue')
        .update({ processed: true, sent: true })
        .eq('id', match.id)
    } catch (err) {
      console.error('Failed to send lead email:', err)
    }
  }
}

export async function runMatchingEngine() {
  const supabase = createAdminClient()

  const { data: businesses } = await supabase
    .from('businesses')
    .select('*')
    .eq('is_active', true)

  if (!businesses?.length) return

  console.log(`Running matching engine for ${businesses.length} businesses`)

  const MATCH_THRESHOLD = 0.65
  let evaluated = 0

  for (const business of businesses) {
    const candidates = businesses.filter(b => b.id !== business.id)

    for (const candidate of candidates) {
      // Pre-filter by industry affinity
      if (business.industry && candidate.industry) {
        const compatible = areIndustriesCompatible(
          business.industry,
          candidate.industry
        )
        if (!compatible) continue
      }

      // Skip if already matched
      const alreadyMatched = await checkAlreadyMatched(business.id, candidate.id)
      if (alreadyMatched) continue

      // Score with Claude
      try {
        const score = await scoreMatch(business, candidate)
        evaluated++

        if (score.score >= MATCH_THRESHOLD && score.send) {
          await supabase.from('match_queue').upsert({
            business_a_id: business.id,
            business_b_id: candidate.id,
            score: score.score,
            match_reason: score.reason,
          })
        }
      } catch (err) {
        console.error(`Failed to score match: ${business.id} <-> ${candidate.id}`, err)
      }
    }
  }

  console.log(`Evaluated ${evaluated} pairs`)

  // Process queue and send emails
  await processMatchQueue()
}

export async function refreshAllLlmsTxt() {
  const supabase = createAdminClient()

  const { data: businesses } = await supabase
    .from('businesses')
    .select('id, url, name, slug, description, services, location, industry, offering_summary')
    .eq('is_active', true)

  if (!businesses?.length) return

  for (const business of businesses) {
    try {
      // Re-scrape and regenerate
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_APP_URL}/api/refresh-llms-txt`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ business_id: business.id }),
        }
      )
      if (!response.ok) {
        console.error(`Failed to refresh llms.txt for ${business.id}`)
      }
    } catch (err) {
      console.error(`Error refreshing ${business.id}:`, err)
    }
  }
}
