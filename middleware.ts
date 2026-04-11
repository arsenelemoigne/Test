import { NextResponse, type NextRequest } from 'next/server'

// Auth is handled in each page/component individually.
// A minimal middleware avoids Edge Runtime compatibility issues
// with newer Supabase key formats.
export function middleware(request: NextRequest) {
  return NextResponse.next()
}

export const config = {
  matcher: [
    '/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)',
  ],
}
