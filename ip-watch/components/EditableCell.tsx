"use client";

import { useState } from "react";
import { formatDateTime } from "@/lib/format";
import type { FieldEdit } from "@/lib/types";

type Props = {
  display: string;
  // Raw value placed in the input when editing starts.
  initial: string;
  edit?: FieldEdit;
  onSave: (raw: string) => Promise<boolean | void> | boolean | void;
  placeholder?: string;
  className?: string;
};

const ROLE_LABEL = { avocat: "l'avocat", client: "le client" } as const;

export function EditableCell({ display, initial, edit, onSave, placeholder, className = "" }: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(initial);
  const [error, setError] = useState(false);

  const commit = async () => {
    if (draft === initial) return setEditing(false);
    if ((await onSave(draft)) === false) return setError(true);
    setEditing(false);
  };

  const title = edit ? `Modifié par ${ROLE_LABEL[edit.by]} le ${formatDateTime(edit.at)} — avant : ${Array.isArray(edit.prev) ? edit.prev.join(", ") : String(edit.prev ?? "—")}` : "Cliquer pour modifier";

  if (editing) {
    return (
      <input
        autoFocus
        className={`w-full rounded border bg-white px-1 py-0.5 text-sm ${error ? "border-red-600" : "border-[color:var(--ink)]"}`}
        value={draft}
        placeholder={placeholder}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit();
          if (e.key === "Escape") setEditing(false);
        }}
      />
    );
  }

  return (
    <button
      type="button"
      onClick={() => {
        setDraft(initial);
        setError(false);
        setEditing(true);
      }}
      title={title}
      className={`w-full rounded text-left hover:bg-[color:var(--faint)] ${className}`}
    >
      {display}
      {edit && <span className="ml-1 inline-block h-1.5 w-1.5 -translate-y-0.5 rounded-full bg-[color:var(--ink)] align-middle" aria-label="modifié" />}
    </button>
  );
}
