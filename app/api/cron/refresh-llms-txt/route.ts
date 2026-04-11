import { NextRequest, NextResponse } from 'next/server'
import { createAdminClient } from '@/lib/supabase/admin'
import OpenAI from 'openai'

function getOpenAI() {
  return new OpenAI({ apiKey: process.env.OPENAI_API_KEY! })
}

export async function GET(request: NextRequest) {
  const authHeader = request.headers.get('authorization')
  const expected = `Bearer ${process.env.CRON_SECRET}`

  if (authHeader !== expected) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const supabase = createAdminClient()

  const { data: businesses } = await supabase
    .from('businesses')
    .select('id, url, name, slug, description, services, location, industry, offering_summary')
    .eq('is_active', true)

  if (!businesses?.length) {
    return NextResponse.json({ success: true, count: 0 })
  }

  let refreshed = 0
  const openai = getOpenAI()
  const today = new Date().toISOString().split('T')[0]

  for (const business of businesses) {
    try {
      const scrapeRes = await fetch(business.url, {
        headers: { 'User-Agent': 'Meshly/1.0 (+https://meshly.com)' },
        signal: AbortSignal.timeout(10000),
      })

      if (!scrapeRes.ok) continue

      const response = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        max_tokens: 800,
        messages: [{
          role: 'user',
          content: `Generate a clean llms.txt for:
Name: ${business.name}
Description: ${business.description}
Services: ${business.services?.join(', ')}
Location: ${business.location}
Slug: ${business.slug}
Offering: ${business.offering_summary}

Format:
# ${business.name}
> ${business.offering_summary}

## About
${business.description}

## Services
${business.services?.map((s: string) => `- ${s}`).join('\n')}

## Location
${business.location}

## Contact
Managed via Meshly network. To connect with this business, reach out through the Meshly platform at https://meshly.com/b/${business.slug}

## Last updated
${today}

# Profile managed by Meshly — meshly.com`,
        }],
      })

      const txt = response.choices[0]?.message?.content || ''

      await supabase
        .from('businesses')
        .update({
          llms_txt_content: txt,
          llms_txt_generated_at: new Date().toISOString(),
        })
        .eq('id', business.id)

      refreshed++
    } catch (err) {
      console.error(`Failed to refresh ${business.id}:`, err)
    }
  }

  return NextResponse.json({ success: true, refreshed, total: businesses.length })
}
