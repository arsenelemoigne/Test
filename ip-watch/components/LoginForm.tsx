"use client";

import { useState } from "react";

export function LoginForm() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (role: "avocat" | "client") => {
    if (!password || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password, role }),
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok || !json.ok) {
        setError(json.error ?? `Connexion impossible (${res.status}).`);
        return;
      }
      window.location.assign("/");
    } catch {
      setError("Serveur injoignable, réessayez.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <form
      className="mt-6 space-y-4"
      onSubmit={(e) => {
        e.preventDefault();
        submit("avocat");
      }}
    >
      <label className="block">
        <span className="label">Mot de passe de démo</span>
        <input
          type="password"
          required
          autoFocus
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="mt-1 w-full border border-black px-2 py-1.5 font-mono text-sm"
        />
      </label>
      {error && <p className="font-mono text-xs text-red-600">{error}</p>}
      <div className="grid grid-cols-2 gap-2">
        <button type="submit" disabled={busy} className="btn-solid justify-center">
          Avocat
        </button>
        <button type="button" disabled={busy} onClick={() => submit("client")} className="btn justify-center">
          Client
        </button>
      </div>
    </form>
  );
}
