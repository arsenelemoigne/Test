import { redirect } from "next/navigation";
import { APP_NAME, FIRM_NAME } from "@/lib/config";
import { currentRole, isConfigured } from "@/lib/auth";
import { LoginForm } from "@/components/LoginForm";

export default function LoginPage() {
  if (currentRole()) redirect("/");
  const configured = isConfigured();

  return (
    <main className="grid min-h-screen place-items-center px-4">
      <div className="w-full max-w-sm border border-black p-6">
        <div className="label">{FIRM_NAME}</div>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">{APP_NAME}</h1>
        <p className="mt-2 text-sm text-neutral-600">Portefeuille de marques et dessins, échéances, veille.</p>

        {configured ? (
          <LoginForm />
        ) : (
          <p className="mt-6 font-mono text-xs text-red-600">
            Variable DEMO_PASSWORD non définie : ajoutez-la dans les réglages Vercel du projet, puis redéployez.
          </p>
        )}
        <p className="mt-6 text-[11px] leading-relaxed text-neutral-500">Démo pédagogique — Projet LAB, EFB. Accès réservé au groupe.</p>
      </div>
    </main>
  );
}
