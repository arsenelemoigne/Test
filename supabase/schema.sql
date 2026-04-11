-- Meshly Database Schema
-- Run this in Supabase SQL editor

-- Core business profile
CREATE TABLE IF NOT EXISTS businesses (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  url TEXT NOT NULL UNIQUE,
  email TEXT,
  slug TEXT UNIQUE,
  user_id UUID REFERENCES auth.users(id),

  -- Scraped automatically
  name TEXT,
  description TEXT,
  location TEXT,
  industry TEXT,
  services TEXT[],
  languages TEXT[],
  raw_scraped_content TEXT,

  -- Generated
  llms_txt_content TEXT,
  llms_txt_generated_at TIMESTAMP WITH TIME ZONE,

  -- User-provided (filled in platform)
  ideal_customer_profile TEXT,
  customer_industries TEXT[],
  customer_size TEXT, -- 'solo', 'small', 'medium', 'large', 'any'
  geographic_scope TEXT, -- 'local', 'national', 'international'
  deal_size TEXT, -- 'small', 'medium', 'large', 'any'
  offering_summary TEXT,
  needs TEXT[],
  inferred_needs TEXT[],
  keywords TEXT[],

  -- Status
  is_active BOOLEAN DEFAULT true,
  profile_completeness INTEGER DEFAULT 0
);

-- Every lead delivered
CREATE TABLE IF NOT EXISTS leads (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  recipient_business_id UUID REFERENCES businesses(id),
  sender_business_id UUID REFERENCES businesses(id),
  match_score FLOAT,
  match_reason TEXT,
  status TEXT DEFAULT 'sent', -- 'sent', 'opened', 'interested', 'declined', 'expired'
  email_sent_at TIMESTAMP WITH TIME ZONE,
  email_opened_at TIMESTAMP WITH TIME ZONE,
  responded_at TIMESTAMP WITH TIME ZONE
);

-- LLM visit log (key feature)
CREATE TABLE IF NOT EXISTS llm_visits (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  visited_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  business_id UUID REFERENCES businesses(id),
  user_agent TEXT,
  bot_name TEXT,
  ip_address TEXT,
  referrer TEXT
);

-- Match queue (processed by cron job)
CREATE TABLE IF NOT EXISTS match_queue (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  business_a_id UUID REFERENCES businesses(id),
  business_b_id UUID REFERENCES businesses(id),
  score FLOAT,
  match_reason TEXT,
  processed BOOLEAN DEFAULT false,
  sent BOOLEAN DEFAULT false,
  UNIQUE(business_a_id, business_b_id)
);

-- Row Level Security

-- Enable RLS
ALTER TABLE businesses ENABLE ROW LEVEL SECURITY;
ALTER TABLE leads ENABLE ROW LEVEL SECURITY;
ALTER TABLE llm_visits ENABLE ROW LEVEL SECURITY;
ALTER TABLE match_queue ENABLE ROW LEVEL SECURITY;

-- Businesses: users can only read/write their own row
CREATE POLICY "Users can view their own business" ON businesses
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can update their own business" ON businesses
  FOR UPDATE USING (auth.uid() = user_id);

-- Allow insert for anon (onboarding flow — user not logged in yet)
CREATE POLICY "Anyone can create a business" ON businesses
  FOR INSERT WITH CHECK (true);

-- Allow public read of basic fields for llms.txt serving
CREATE POLICY "Public can read llms_txt_content by slug" ON businesses
  FOR SELECT USING (true);

-- Leads: users can only read leads where they are the recipient
CREATE POLICY "Users can view their own leads" ON leads
  FOR SELECT USING (
    recipient_business_id IN (
      SELECT id FROM businesses WHERE user_id = auth.uid()
    )
  );

-- LLM visits: users can only read visits for their own business
CREATE POLICY "Users can view their own LLM visits" ON llm_visits
  FOR SELECT USING (
    business_id IN (
      SELECT id FROM businesses WHERE user_id = auth.uid()
    )
  );

-- Allow insert for llm_visits (public — no auth needed for tracking)
CREATE POLICY "Anyone can insert llm_visits" ON llm_visits
  FOR INSERT WITH CHECK (true);

-- Match queue: server-side only (service role bypasses RLS)
-- No client-side policies needed

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_businesses_slug ON businesses(slug);
CREATE INDEX IF NOT EXISTS idx_businesses_user_id ON businesses(user_id);
CREATE INDEX IF NOT EXISTS idx_businesses_is_active ON businesses(is_active);
CREATE INDEX IF NOT EXISTS idx_leads_recipient ON leads(recipient_business_id);
CREATE INDEX IF NOT EXISTS idx_leads_status ON leads(status);
CREATE INDEX IF NOT EXISTS idx_llm_visits_business ON llm_visits(business_id);
CREATE INDEX IF NOT EXISTS idx_llm_visits_bot ON llm_visits(bot_name);
CREATE INDEX IF NOT EXISTS idx_match_queue_processed ON match_queue(processed, sent);
