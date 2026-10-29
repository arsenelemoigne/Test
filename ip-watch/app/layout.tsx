import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import { APP_NAME } from "@/lib/config";
import "./globals.css";

export const metadata: Metadata = {
  title: APP_NAME,
  description: "Démo — suivi d'un portefeuille de marques et dessins, échéances et veille.",
  robots: { index: false, follow: false },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr" className={`${GeistSans.variable} ${GeistMono.variable}`}>
      <body className="min-h-screen font-sans antialiased">{children}</body>
    </html>
  );
}
