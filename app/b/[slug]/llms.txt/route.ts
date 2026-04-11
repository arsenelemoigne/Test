import { NextRequest, NextResponse } from 'next/server'
import { createAdminClient } from '@/lib/supabase/admin'

const LLM_BOTS: Record<string, string> = {
  'GPTBot': 'OpenAI',
  'ChatGPT-User': 'ChatGPT',
  'ClaudeBot': 'Anthropic Claude',
  'Claude-Web': 'Anthropic Claude',
  'anthropic-ai': 'Anthropic',
  'PerplexityBot': 'Perplexity',
  'Googlebot': 'Google',
  'Google-Extended': 'Google Gemini',
  'YouBot': 'You.com',
  'cohere-ai': 'Cohere',
  'meta-externalagent': 'Meta AI',
  'Applebot': 'Apple',
  'Bytespider': 'ByteDance',
}

export async function GET(
  request: NextRequest,
  { params }: { params: { slug: string } }
) {
  const { slug } = params
  const supabase = createAdminClient()

  const { data: business, error } = await supabase
    .from('businesses')
    .select('id, llms_txt_content')
    .eq('slug', slug)
    .single()

  if (error || !business) {
    return new NextResponse('Not found', { status: 404 })
  }

  // Detect LLM bot
  const userAgent = request.headers.get('user-agent') || ''
  const ip = request.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || ''
  const referrer = request.headers.get('referer') || ''

  let detectedBot: string | null = null
  for (const [botKey, botName] of Object.entries(LLM_BOTS)) {
    if (userAgent.toLowerCase().includes(botKey.toLowerCase())) {
      detectedBot = botName
      break
    }
  }

  // Log the visit (fire-and-forget)
  void supabase.from('llm_visits').insert({
    business_id: business.id,
    user_agent: userAgent,
    bot_name: detectedBot,
    ip_address: ip,
    referrer: referrer,
  })

  return new NextResponse(business.llms_txt_content || '', {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': 'public, max-age=3600',
      'X-Robots-Tag': 'noindex',
    },
  })
}
