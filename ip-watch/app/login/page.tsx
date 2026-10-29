import { redirect } from "next/navigation";
import { APP_NAME, FIRM_NAME } from "@/lib/config";
import { currentRole, isConfigured } from "@/lib/auth";
import { LoginForm } from "@/components/LoginForm";
import { RoughBox, Scribble } from "@/components/RoughBox";

export default function LoginPage() {
  if (currentRole()) redirect("/");
  const configured = isConfigured();

  return (
    <main className="grid min-h-screen place-items-center px-4">
      <RoughBox seed={2} className="w-full max-w-sm">
        <div className="p-7">
          <div className="label">{FIRM_NAME}</div>
          <h1 className="hand mt-1 text-[28px] font-semibold leading-none">{APP_NAME}</h1>
          <Scribble width={120} className="mt-1" />
          <p className="mt-3 text-[13px] text-[color:var(--pencil)]">Marques, échéances, contrefaçons — et un assistant qui cite ses sources.</p>

          {configured ? (
            <LoginForm />
          ) : (
            <p className="hand mt-6 text-[13px] text-red-700">Variable DEMO_PASSWORD non définie dans Railway.</p>
          )}
          <p className="mt-6 text-[11px] leading-relaxed text-[color:var(--pencil)]">Démo pédagogique — Projet LAB, EFB. Accès réservé au groupe.</p>
        </div>
      </RoughBox>
    </main>
  );
}
