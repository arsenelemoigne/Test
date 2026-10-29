// Tiny Markdown subset used by the assistant: paragraphs, bullets, quotes, **bold**, *italic*, [links](url).
export type Inline = { kind: "text" | "bold" | "italic" | "code"; text: string } | { kind: "link"; text: string; href: string };
export type Block = { kind: "p" | "quote"; inlines: Inline[] } | { kind: "list"; ordered: boolean; items: Inline[][] };

const INLINE = /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)|\*\*([^*]+)\*\*|`([^`]+)`|\*([^*\s][^*]*)\*/g;

export function parseInline(text: string): Inline[] {
  const out: Inline[] = [];
  let last = 0;
  for (const m of Array.from(text.matchAll(INLINE))) {
    if (m.index! > last) out.push({ kind: "text", text: text.slice(last, m.index) });
    if (m[1]) out.push({ kind: "link", text: m[1], href: m[2] });
    else if (m[3]) out.push({ kind: "bold", text: m[3] });
    else if (m[4]) out.push({ kind: "code", text: m[4] });
    else if (m[5]) out.push({ kind: "italic", text: m[5] });
    last = m.index! + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}

export function parseBlocks(md: string): Block[] {
  const blocks: Block[] = [];
  let para: string[] = [];
  let list: { ordered: boolean; items: Inline[][] } | null = null;
  const flushPara = () => {
    if (para.length) blocks.push({ kind: "p", inlines: parseInline(para.join(" ")) });
    para = [];
  };
  const flushList = () => {
    if (list) blocks.push({ kind: "list", ...list });
    list = null;
  };
  for (const raw of md.split(/\r?\n/)) {
    const line = raw.trim();
    const bullet = line.match(/^[-*•]\s+(.*)$/);
    const numbered = line.match(/^\d+[.)]\s+(.*)$/);
    if (!line) {
      flushPara();
      flushList();
    } else if (bullet || numbered) {
      flushPara();
      const ordered = Boolean(numbered);
      if (!list || list.ordered !== ordered) {
        flushList();
        list = { ordered, items: [] };
      }
      list.items.push(parseInline((bullet ?? numbered)![1]));
    } else if (line.startsWith(">")) {
      flushPara();
      flushList();
      blocks.push({ kind: "quote", inlines: parseInline(line.replace(/^>\s?/, "")) });
    } else {
      flushList();
      para.push(line.replace(/^#{1,6}\s+/, ""));
    }
  }
  flushPara();
  flushList();
  return blocks;
}

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

function inlineHtml(inlines: Inline[]): string {
  return inlines
    .map((i) => {
      if (i.kind === "link") return `<a href="${esc(i.href)}">${esc(i.text)}</a>`;
      if (i.kind === "bold") return `<strong>${esc(i.text)}</strong>`;
      if (i.kind === "italic") return `<em>${esc(i.text)}</em>`;
      if (i.kind === "code") return `<code>${esc(i.text)}</code>`;
      return esc(i.text);
    })
    .join("");
}

export function toHtml(md: string): string {
  return parseBlocks(md)
    .map((b) => {
      if (b.kind === "list") return `<${b.ordered ? "ol" : "ul"}>${b.items.map((it) => `<li>${inlineHtml(it)}</li>`).join("")}</${b.ordered ? "ol" : "ul"}>`;
      if (b.kind === "quote") return `<blockquote style="margin:0 0 8px;padding-left:10px;border-left:2px solid #999;color:#444">${inlineHtml(b.inlines)}</blockquote>`;
      return `<p style="margin:0 0 10px">${inlineHtml(b.inlines)}</p>`;
    })
    .join("");
}
