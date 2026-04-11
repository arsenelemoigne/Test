// Industry affinity table — pre-filter before Claude scoring
// This eliminates ~95% of pairs before hitting the Claude API

export const INDUSTRY_AFFINITY: Record<string, string[]> = {
  'Hair salon': [
    'Cosmetics supplier', 'Beauty distributor', 'Booking software',
    'Marketing agency', 'Photography', 'Wedding planner', 'Fashion',
    'Social media agency', 'E-commerce', 'Retail',
  ],
  'Restaurant': [
    'Food supplier', 'Beverage supplier', 'Delivery platform',
    'Accounting software', 'POS software', 'Marketing agency',
    'Photography', 'Interior design', 'Hospitality',
  ],
  'Marketing agency': [
    'E-commerce', 'Retail', 'Tech/SaaS', 'Finance', 'Health',
    'Food & Beverage', 'Construction', 'Real estate', 'Education',
    'Print supplier', 'Developer', 'Photographer', 'Hospitality',
    'Manufacturing', 'Creative/Agency',
  ],
  'Software': [
    'Tech/SaaS', 'E-commerce', 'Finance', 'Health', 'Education',
    'Marketing agency', 'Manufacturing', 'Retail', 'Real estate',
  ],
  'Tech/SaaS': [
    'E-commerce', 'Finance', 'Health', 'Education', 'Marketing agency',
    'Manufacturing', 'Retail', 'Real estate', 'Construction', 'Hospitality',
  ],
  'Photography': [
    'Wedding planner', 'Marketing agency', 'E-commerce', 'Fashion',
    'Hair salon', 'Restaurant', 'Real estate', 'Events',
  ],
  'Wedding planner': [
    'Photography', 'Hair salon', 'Fashion', 'Catering', 'Florist',
    'Music', 'Venue', 'Hospitality',
  ],
  'E-commerce': [
    'Marketing agency', 'Logistics', 'Tech/SaaS', 'Photography',
    'Retail', 'Manufacturing', 'Wholesale',
  ],
  'Real estate': [
    'Photography', 'Interior design', 'Construction', 'Finance',
    'Marketing agency', 'Legal',
  ],
  'Construction': [
    'Architecture', 'Real estate', 'Materials supplier', 'Engineering',
    'Interior design', 'Finance',
  ],
  'Health': [
    'Tech/SaaS', 'Medical supplier', 'Marketing agency', 'Finance',
    'Education', 'Insurance',
  ],
  'Finance': [
    'Tech/SaaS', 'Legal', 'Real estate', 'Insurance', 'Education',
    'Marketing agency',
  ],
  'Education': [
    'Tech/SaaS', 'Marketing agency', 'Finance', 'Publishing', 'Health',
  ],
}

export function areIndustriesCompatible(industryA: string, industryB: string): boolean {
  // Check if they are the same industry (potential competitor but also potential partner)
  if (industryA === industryB) return true

  const affinityA = INDUSTRY_AFFINITY[industryA] || []
  const affinityB = INDUSTRY_AFFINITY[industryB] || []

  // Check if B's industry appears in A's affinity list
  if (affinityA.some(a => industryB.toLowerCase().includes(a.toLowerCase()) ||
    a.toLowerCase().includes(industryB.toLowerCase()))) return true

  // Check if A's industry appears in B's affinity list
  if (affinityB.some(b => industryA.toLowerCase().includes(b.toLowerCase()) ||
    b.toLowerCase().includes(industryA.toLowerCase()))) return true

  // If either industry isn't in our table, allow it through (better to over-match)
  if (!INDUSTRY_AFFINITY[industryA] || !INDUSTRY_AFFINITY[industryB]) return true

  return false
}
