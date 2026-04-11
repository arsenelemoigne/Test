import { Business } from '@/lib/types'

export function calculateProfileCompleteness(business: Partial<Business>): number {
  let score = 0

  // Name + description filled: +20
  if (business.name && business.description) score += 20

  // Services filled: +15
  if (business.services && business.services.length > 0) score += 15

  // Ideal customer profile filled: +20
  if (business.ideal_customer_profile) score += 20

  // Target industries selected: +15
  if (business.customer_industries && business.customer_industries.length > 0) score += 15

  // Needs/looking for selected: +20
  if (business.needs && business.needs.length > 0) score += 20

  // Keywords added: +10
  if (business.keywords && business.keywords.length > 0) score += 10

  return Math.min(score, 100)
}
