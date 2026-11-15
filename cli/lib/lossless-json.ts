/**
 * JSON that keeps 64-bit integers exact, for SQL results only.
 *
 * `JSON.parse` turns -8580166607628940136 (a `queryid`) into the nearest
 * double, -8580166607628940000, which then matches nothing. Here an integer
 * literal outside ±(2^53−1) becomes a `bigint` on parse and is written back as
 * the same bare JSON number on stringify; everything else is exactly what
 * `JSON.parse` / `JSON.stringify` give.
 *
 * A text pass, not `JSON.parse` source access or `JSON.rawJSON`: the CLI also
 * ships for Node 18+ (`bun build --target node`), which has neither.
 */

// A per-call marker: the store's own strings cannot carry it by accident.
const marker = () => `pgai-int-${Math.random().toString(36).slice(2)}:`;

// A JSON number at the cursor, and whether it has a fraction or an exponent.
const NUMBER = /-?(?:0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?/y;

export function parseLossless(text: string): unknown {
  // A safe integer has at most 16 digits; no run of 16 means nothing to keep.
  if (!/\d{16}/.test(text)) return JSON.parse(text);
  const tag = marker();
  let out = "";
  let last = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      // Skip the string whole: digits inside it stay a string.
      for (i++; i < text.length && text[i] !== '"'; i++) if (text[i] === "\\") i++;
    } else if (c === "-" || (c >= "0" && c <= "9")) {
      NUMBER.lastIndex = i;
      const m = NUMBER.exec(text);
      if (!m) continue; // not a number: let JSON.parse report it
      const end = i + m[0].length;
      if (!m[1] && !m[2] && !Number.isSafeInteger(Number(m[0]))) {
        out += text.slice(last, i) + `"${tag}${m[0]}"`;
        last = end;
      }
      i = end - 1;
    }
  }
  if (!last) return JSON.parse(text);
  return JSON.parse(out + text.slice(last), (_k, v) =>
    typeof v === "string" && v.startsWith(tag) ? BigInt(v.slice(tag.length)) : v,
  );
}

export function stringifyLossless(value: unknown, indent?: number): string {
  const tag = marker();
  const text = JSON.stringify(value, (_k, v) => (typeof v === "bigint" ? `${tag}${v}` : v), indent);
  return text.replace(new RegExp(`"${tag}(-?\\d+)"`, "g"), "$1");
}
