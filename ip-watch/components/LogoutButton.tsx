"use client";

export function LogoutButton() {
  return (
    <button
      type="button"
      className="btn-quiet"
      onClick={async () => {
        await fetch("/api/logout", { method: "POST" }).catch(() => null);
        window.location.assign("/login");
      }}
    >
      quitter
    </button>
  );
}
