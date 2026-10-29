import { SESSION_COOKIE } from "@/lib/auth";
import { seeOther } from "@/lib/http";

export async function POST() {
  const response = seeOther("/login");
  response.cookies.delete(SESSION_COOKIE);
  return response;
}
