// INPI "API diffusion" client. Login needs a cookie jar and an anti-CSRF token that changes on each response.
const BASE = "https://api-gateway.inpi.fr";

export class InpiSession {
  private jar = new Map<string, string>();

  private remember(res: Response) {
    for (const line of res.headers.getSetCookie()) {
      const [pair] = line.split(";");
      const i = pair.indexOf("=");
      if (i > 0) this.jar.set(pair.slice(0, i).trim(), pair.slice(i + 1).trim());
    }
  }

  private headers(json: boolean): Record<string, string> {
    const h: Record<string, string> = { Accept: "application/json" };
    if (json) h["Content-Type"] = "application/json";
    if (this.jar.size) h.Cookie = Array.from(this.jar.entries()).map(([k, v]) => `${k}=${v}`).join("; ");
    const xsrf = this.jar.get("XSRF-TOKEN");
    if (xsrf) h["X-XSRF-TOKEN"] = decodeURIComponent(xsrf);
    return h;
  }

  async request(method: "GET" | "POST", path: string, body?: unknown): Promise<{ status: number; data: unknown }> {
    const res = await fetch(`${BASE}${path}`, {
      method,
      headers: this.headers(body !== undefined),
      body: body === undefined ? undefined : JSON.stringify(body),
      redirect: "manual",
    });
    this.remember(res);
    const text = await res.text();
    let data: unknown = text;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      // Keep non-JSON bodies (HTML error pages) as text.
    }
    return { status: res.status, data };
  }

  // Returns the login HTTP status; the body is discarded because it may contain tokens.
  async login(username: string, password: string): Promise<number> {
    await this.request("GET", "/services/uaa/api/authenticate");
    const { status } = await this.request("POST", "/auth/login", { username, password, rememberMe: true });
    return status;
  }
}

// Number of hits in a search page, whatever the envelope looks like.
export function countHits(data: unknown): number | null {
  if (Array.isArray(data)) return data.length;
  if (data && typeof data === "object") {
    for (const value of Object.values(data as Record<string, unknown>)) {
      if (Array.isArray(value)) return value.length;
    }
  }
  return null;
}
