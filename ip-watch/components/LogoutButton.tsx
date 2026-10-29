"use client";

export function LogoutButton() {
  return (
    <button
      type="button"
      className="btn"
      onClick={async () => {
        await fetch("/api/logout", { method: "POST" }).catch(() => null);
        window.location.assign("/login");
      }}
    >
      Quitter
    </button>
  );
}
