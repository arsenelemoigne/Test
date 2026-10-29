import { redirect } from "next/navigation";
import { currentClient } from "@/lib/data";
import * as store from "@/lib/store";
import { MarksTable } from "@/components/MarksTable";

export const dynamic = "force-dynamic";

export default async function MarksPage() {
  const client = await currentClient();
  if (!client) redirect("/");
  const marks = await store.listMarks(client.id);
  return <MarksTable key={client.id} marks={marks} />;
}
