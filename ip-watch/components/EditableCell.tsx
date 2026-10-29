"use client";

import { useState } from "react";
import type { FieldEdit } from "./EditsProvider";
import { formatDateTime } from "@/lib/format";

type Props = {
  display: string;
  // Raw value placed in the input when editing starts.
  initial: string;
  edit?: FieldEdit;
  onSave: (raw: string) => boolean | void;
  options?: { value: string; label: string }[];
  placeholder?: string;
  className?: string;
};

const ROLE_LABEL = { avocat: "l'avocat", client: "le client" } as const;

export function EditableCell({ display, initial, edit, onSave, options, placeholder, className = "" }: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(initial);
  const [error, setError] = useState(false);

  const start = () => {
    setDraft(initial);
    setError(false);
    setEditing(true);
  };

  const commit = (value = draft) => {
    if (onSave(value) === false) {
      setError(true);
      return;
    }
    setEditing(false);
  };

  const title = edit ? `Modifié par ${ROLE_LABEL[edit.by]} le ${formatDateTime(edit.at)} — avant : ${String(edit.prev ?? "—")}` : "Cliquer pour modifier";

  if (editing) {
    if (options) {
      return (
        <select
          autoFocus
          className="w-full border border-black bg-white px-1 py-0.5 text-sm"
          value={draft}
          onChange={(e) => commit(e.target.value)}
          onBlur={() => setEditing(false)}
        >
          {options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      );
    }
    return (
      <input
        autoFocus
        className={`w-full border bg-white px-1 py-0.5 text-sm ${error ? "border-red-600" : "border-black"}`}
        value={draft}
        placeholder={placeholder}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => commit()}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit();
          if (e.key === "Escape") setEditing(false);
        }}
      />
    );
  }

  return (
    <button type="button" onClick={start} title={title} className={`group w-full text-left hover:bg-neutral-100 ${className}`}>
      {display}
      {edit && <span className="ml-1 inline-block h-1.5 w-1.5 translate-y-[-2px] rounded-full bg-black align-middle" aria-label="modifié" />}
    </button>
  );
}
