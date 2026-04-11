import Link from 'next/link'

export default function ThanksFeedbackPage() {
  return (
    <div className="min-h-screen bg-white flex flex-col items-center justify-center px-6 text-center">
      <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mb-6">
        <span className="text-gray-500 text-xl">✓</span>
      </div>
      <h1 className="text-2xl font-bold text-gray-900 mb-2">Thanks for the feedback.</h1>
      <p className="text-gray-500 mb-6">We won&apos;t send you this contact again.</p>
      <Link href="/leads" className="text-sm text-gray-900 underline">
        View all your leads →
      </Link>
    </div>
  )
}
