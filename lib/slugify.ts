import slugifyLib from 'slugify'
import { createAdminClient } from '@/lib/supabase/admin'

export function slugifyName(name: string): string {
  return slugifyLib(name, {
    lower: true,
    strict: true,
    trim: true,
  })
}

export async function generateUniqueSlug(name: string): Promise<string> {
  const supabase = createAdminClient()
  const base = slugifyName(name)

  // Check if base slug exists
  const { data } = await supabase
    .from('businesses')
    .select('slug')
    .eq('slug', base)
    .single()

  if (!data) return base

  // Try appending numbers
  for (let i = 2; i <= 99; i++) {
    const candidate = `${base}-${i}`
    const { data: existing } = await supabase
      .from('businesses')
      .select('slug')
      .eq('slug', candidate)
      .single()

    if (!existing) return candidate
  }

  // Fallback with random suffix
  return `${base}-${Math.random().toString(36).slice(2, 7)}`
}
