export interface Business {
  id: string
  created_at: string
  url: string
  email: string | null
  slug: string | null
  user_id: string | null

  // Scraped
  name: string | null
  description: string | null
  location: string | null
  industry: string | null
  services: string[] | null
  languages: string[] | null
  raw_scraped_content: string | null

  // Generated
  llms_txt_content: string | null
  llms_txt_generated_at: string | null

  // User-provided
  ideal_customer_profile: string | null
  customer_industries: string[] | null
  customer_size: string | null
  geographic_scope: string | null
  deal_size: string | null
  offering_summary: string | null
  needs: string[] | null
  inferred_needs: string[] | null
  keywords: string[] | null

  // Status
  is_active: boolean
  profile_completeness: number
}

export interface Lead {
  id: string
  created_at: string
  recipient_business_id: string
  sender_business_id: string
  match_score: number
  match_reason: string | null
  status: 'sent' | 'opened' | 'interested' | 'declined' | 'expired'
  email_sent_at: string | null
  email_opened_at: string | null
  responded_at: string | null
  // Joined
  sender?: Business
  recipient?: Business
}

export interface LlmVisit {
  id: string
  visited_at: string
  business_id: string
  user_agent: string | null
  bot_name: string | null
  ip_address: string | null
  referrer: string | null
}

export interface MatchQueueItem {
  id: string
  created_at: string
  business_a_id: string
  business_b_id: string
  score: number
  match_reason: string | null
  processed: boolean
  sent: boolean
}

export interface AnalyzedBusiness {
  name: string
  description: string
  industry: string
  location: string
  services: string[]
  languages: string[]
  inferred_needs: string[]
  keywords: string[]
  offering_summary: string
}
