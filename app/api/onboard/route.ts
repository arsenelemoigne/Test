import { NextRequest, NextResponse } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'
import * as cheerio from 'cheerio'
import { createAdminClient } from '@/lib/supabase/admin'
import { generateUniqueSlug } from '@/lib/slugify'
import { AnalyzedBusiness } from '@/lib/types'

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY!,
})

// Rate limiting (in-memory, simple MVP version)
const submissionsByIP: Map<string, { count: number; resetAt: number }> = new Map()

function checkRateLimit(ip: string): boolean {
  const now = Date.now()
  const hourMs = 60 * 60 * 1000
  const entry = submissionsByIP.get(ip)

  if (!entry || entry.resetAt < now) {
    submissionsByIP.set(ip, { count: 1, resetAt: now + hourMs })
    return true
  }

  if (entry.count >= 3) return false

  entry.count++
  return true
}

function isPrivateURL(url: string): boolean {
  try {
    const parsed = new URL(url)
    const hostname = parsed.hostname

    // Block localhost and private IPs
    const blocked = [
      'localhost', '127.0.0.1', '0.0.0.0', '::1',
      '10.', '172.16.', '172.17.', '172.18.', '172.19.',
      '172.20.', '172.21.', '172.22.', '172.23.', '172.24.',
      '172.25.', '172.26.', '172.27.', '172.28.', '172.29.',
      '172.30.', '172.31.', '192.168.',
    ]
    return blocked.some(b => hostname.startsWith(b) || hostname === b)
  } catch {
    return true
  }
}

async function scrapeWithFirecrawl(url: string): Promise<string | null> {
  const apiKey = process.env.FIRECRAWL_API_KEY
  if (!apiKey) return null

  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const firecrawl = await import('@mendable/firecrawl-js') as any
    const FirecrawlApp = firecrawl.default || firecrawl.FirecrawlApp || firecrawl.Firecrawl
    const app = new FirecrawlApp({ apiKey })
    const result = await app.scrapeUrl(url, {
      formats: ['markdown'],
    }) as { success: boolean; markdown?: string }
    if (result.success && result.markdown) {
      return result.markdown.slice(0, 10000)
    }
    return null
  } catch (err) {
    console.error('Firecrawl failed:', err)
    return null
  }
}

async function scrapeWithCheerio(url: string): Promise<string | null> {
  try {
    const response = await fetch(url, {
      headers: { 'User-Agent': 'Meshly/1.0 (+https://meshly.com)' },
      signal: AbortSignal.timeout(10000),
    })
    const html = await response.text()
    const $ = cheerio.load(html)

    // Remove scripts, styles, nav, footer
    $('script, style, nav, footer, header, .cookie-banner').remove()

    const title = $('title').text().trim()
    const metaDesc = $('meta[name="description"]').attr('content') || ''
    const h1s = $('h1').map((_, el) => $(el).text().trim()).get().join(' | ')
    const body = $('body').text().replace(/\s+/g, ' ').trim().slice(0, 8000)

    return `Title: ${title}\nMeta: ${metaDesc}\nH1s: ${h1s}\n\nContent:\n${body}`
  } catch (err) {
    console.error('Cheerio scrape failed:', err)
    return null
  }
}

async function analyzeWithClaude(rawContent: string): Promise<AnalyzedBusiness> {
  const response = await anthropic.messages.create({
    model: 'claude-haiku-4-5',
    max_tokens: 1000,
    system: 'You are a business analyst. Extract structured information from website content. Return ONLY valid JSON, no markdown, no explanation.',
    messages: [
      {
        role: 'user',
        content: `Analyze this website content and return:
{
  "name": "business name",
  "description": "2-sentence description of what they do",
  "industry": "single industry label (e.g. 'Hair salon', 'Marketing agency', 'Software')",
  "location": "city, country if found",
  "services": ["service1", "service2", "service3"],
  "languages": ["fr", "en"],
  "inferred_needs": ["thing they likely need 1", "thing they likely need 2"],
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "offering_summary": "One sentence: what they sell and to whom"
}

IMPORTANT on inferred_needs: Based on their industry and services, infer what kinds of B2B products or services they are LIKELY to need. Examples:
- Hair salon → ["hair product suppliers", "booking software", "social media agency"]
- Marketing agency → ["SaaS clients", "print providers", "freelance developers"]
- Restaurant → ["food suppliers", "delivery platforms", "accounting software"]
This inference is critical for matching before the user tells us themselves.

Website content: ${rawContent}`,
      },
    ],
  })

  const text = response.content[0].type === 'text' ? response.content[0].text : ''
  return JSON.parse(text) as AnalyzedBusiness
}

async function generateLlmsTxt(
  business: AnalyzedBusiness,
  slug: string
): Promise<string> {
  const today = new Date().toISOString().split('T')[0]

  const response = await anthropic.messages.create({
    model: 'claude-haiku-4-5',
    max_tokens: 800,
    system: 'Generate a clean llms.txt file for a business. Use this exact format. No extra commentary.',
    messages: [
      {
        role: 'user',
        content: `Generate an llms.txt file for this business:

Name: ${business.name}
Description: ${business.description}
Services: ${business.services.join(', ')}
Location: ${business.location}
Industry: ${business.industry}
Offering: ${business.offering_summary}
Slug: ${slug}

Use this format:
# [name]
> [offering_summary]

## About
[description]

## Services
[services as bullet list with - prefix]

## Location
[location]

## Ideal clients
[inferred from industry — who do they typically serve, 1-2 sentences]

## Contact
Managed via Meshly network. To connect with this business, reach out through the Meshly platform at https://meshly.com/b/[slug]

## Last updated
${today}

# Profile managed by Meshly — meshly.com`,
      },
    ],
  })

  const text = response.content[0].type === 'text' ? response.content[0].text : ''
  return text
}

export async function POST(request: NextRequest) {
  const ip = request.headers.get('x-forwarded-for') || 'unknown'

  // Rate limiting
  if (!checkRateLimit(ip)) {
    return NextResponse.json(
      { error: 'Too many requests. Please try again later.' },
      { status: 429 }
    )
  }

  const body = await request.json()
  const { url } = body

  // Validate URL
  let parsedUrl: URL
  try {
    parsedUrl = new URL(url)
    if (!['http:', 'https:'].includes(parsedUrl.protocol)) {
      throw new Error('Invalid protocol')
    }
  } catch {
    return NextResponse.json({ error: 'Invalid URL format.' }, { status: 400 })
  }

  if (isPrivateURL(url)) {
    return NextResponse.json({ error: 'URL not allowed.' }, { status: 400 })
  }

  // Check if URL already exists
  const supabase = createAdminClient()
  const { data: existing } = await supabase
    .from('businesses')
    .select('id, slug, name')
    .eq('url', url)
    .single()

  if (existing) {
    return NextResponse.json({
      business_id: existing.id,
      slug: existing.slug,
      name: existing.name,
    })
  }

  // Check URL is reachable
  try {
    const headRes = await fetch(url, {
      method: 'HEAD',
      signal: AbortSignal.timeout(5000),
    })
    if (!headRes.ok && headRes.status >= 500) {
      return NextResponse.json({ error: 'URL is not reachable.' }, { status: 400 })
    }
  } catch {
    return NextResponse.json({ error: 'URL is not reachable.' }, { status: 400 })
  }

  // 1. Scrape
  let rawContent = await scrapeWithFirecrawl(url)
  if (!rawContent) {
    rawContent = await scrapeWithCheerio(url)
  }

  if (!rawContent) {
    return NextResponse.json(
      { error: 'Could not read website content.' },
      { status: 422 }
    )
  }

  // 2. AI Analysis
  let analyzed: AnalyzedBusiness
  try {
    analyzed = await analyzeWithClaude(rawContent)
  } catch (err) {
    console.error('AI analysis failed:', err)
    return NextResponse.json(
      { error: 'Failed to analyze website.' },
      { status: 500 }
    )
  }

  // 3. Generate slug
  const slug = await generateUniqueSlug(analyzed.name || parsedUrl.hostname)

  // 4. Generate llms.txt
  let llmsTxtContent: string
  try {
    llmsTxtContent = await generateLlmsTxt(analyzed, slug)
  } catch {
    // Fallback: generate manually without AI
    const today = new Date().toISOString().split('T')[0]
    llmsTxtContent = `# ${analyzed.name}
> ${analyzed.offering_summary}

## About
${analyzed.description}

## Services
${analyzed.services.map(s => `- ${s}`).join('\n')}

## Location
${analyzed.location}

## Contact
Managed via Meshly network. To connect with this business, reach out through the Meshly platform at https://meshly.com/b/${slug}

## Last updated
${today}

# Profile managed by Meshly — meshly.com`
  }

  // 5. Save to database
  const { data: business, error } = await supabase
    .from('businesses')
    .insert({
      url,
      slug,
      name: analyzed.name,
      description: analyzed.description,
      industry: analyzed.industry,
      location: analyzed.location,
      services: analyzed.services,
      languages: analyzed.languages,
      inferred_needs: analyzed.inferred_needs,
      keywords: analyzed.keywords,
      offering_summary: analyzed.offering_summary,
      raw_scraped_content: rawContent,
      llms_txt_content: llmsTxtContent,
      llms_txt_generated_at: new Date().toISOString(),
    })
    .select('id, slug, name')
    .single()

  if (error || !business) {
    console.error('DB insert failed:', error)
    return NextResponse.json(
      { error: 'Failed to save business profile.' },
      { status: 500 }
    )
  }

  return NextResponse.json({
    business_id: business.id,
    slug: business.slug,
    name: business.name,
  })
}
