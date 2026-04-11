export default function ThanksPage() {
  return (
    <div className="min-h-screen bg-white flex flex-col">
      <header className="px-6 py-5 max-w-5xl mx-auto w-full">
        <span className="text-xl font-bold tracking-tight">Meshly</span>
      </header>

      <main className="flex-1 flex flex-col items-center justify-center px-6 text-center">
        <div className="w-16 h-16 bg-gray-900 rounded-full flex items-center justify-center mb-8">
          <span className="text-white text-2xl">✓</span>
        </div>

        <h1 className="text-3xl font-bold tracking-tight text-gray-900 mb-3">
          You&apos;re in the network.
        </h1>
        <p className="text-gray-500 text-lg mb-2">
          We&apos;ll email you when we find a relevant contact.
        </p>
        <p className="text-gray-400 text-base mb-10">
          In the meantime, you can strengthen your profile to get better matches.
        </p>

        <a
          href="/profile"
          className="px-8 py-4 bg-gray-900 text-white font-semibold rounded-xl hover:bg-gray-700 transition-colors text-base"
        >
          Complete my profile →
        </a>
      </main>
    </div>
  )
}
