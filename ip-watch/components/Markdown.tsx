import { Fragment } from "react";
import { parseBlocks, type Inline } from "@/lib/markdown";

function Inlines({ items }: { items: Inline[] }) {
  return (
    <>
      {items.map((i, k) => {
        if (i.kind === "link")
          return (
            <a key={k} href={i.href} target="_blank" rel="noreferrer" className="underline decoration-[color:var(--pencil)] underline-offset-2 hover:decoration-[color:var(--ink)]">
              {i.text}
            </a>
          );
        if (i.kind === "bold") return <strong key={k} className="font-semibold">{i.text}</strong>;
        if (i.kind === "italic") return <em key={k}>{i.text}</em>;
        if (i.kind === "code") return <code key={k} className="font-mono text-[12px]">{i.text}</code>;
        return <Fragment key={k}>{i.text}</Fragment>;
      })}
    </>
  );
}

export function Markdown({ text }: { text: string }) {
  return (
    <div className="space-y-2 text-[13.5px] leading-relaxed">
      {parseBlocks(text).map((b, k) => {
        if (b.kind === "list") {
          const List = b.ordered ? "ol" : "ul";
          return (
            <List key={k} className={`${b.ordered ? "list-decimal" : "list-disc"} space-y-1 pl-5`}>
              {b.items.map((it, j) => (
                <li key={j}>
                  <Inlines items={it} />
                </li>
              ))}
            </List>
          );
        }
        if (b.kind === "quote")
          return (
            <blockquote key={k} className="border-l-2 border-[color:var(--ink)] pl-3 italic text-[color:var(--pencil)]">
              <Inlines items={b.inlines} />
            </blockquote>
          );
        return (
          <p key={k}>
            <Inlines items={b.inlines} />
          </p>
        );
      })}
    </div>
  );
}
