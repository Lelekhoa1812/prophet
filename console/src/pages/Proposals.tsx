import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useApi } from "@/lib/api";
import { useApp } from "@/lib/context";
import { ProposalCard, AGENT_LABEL } from "@/components/ProposalCard";
import { Button, Card, CardContent, EmptyState, InlineAlert, PageHeader, Select, Skeleton } from "@/components/ui";
import { money } from "@/lib/utils";

/** Review queue for every proposal in the dataset. Filters persist in the URL-free state for this session. */
export function Proposals() {
  const { dataset } = useApp();
  const [agent, setAgent] = useState("");
  const [status, setStatus] = useState("open");
  const [sort, setSort] = useState("dollars");
  const q = `/proposals?dataset=${dataset}&limit=60&sort=${sort}${agent ? `&agent=${agent}` : ""}${status ? `&status=${status}` : ""}`;
  const { data, error, loading, reload } = useApi<any[]>(q);
  return (
    <>
      <PageHeader eyebrow="Review queue" title="Proposals waiting for a decision"
        description="Ranked by money at stake. Each one can be approved, adjusted or dismissed. Dismissals teach the agent, and every lesson can be undone."
        actions={<>
          <Select aria-label="Agent" value={agent} onChange={(e) => setAgent(e.target.value)} className="h-8 text-xs">
            <option value="">All agents</option>{Object.entries(AGENT_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </Select>
          <Select aria-label="Status" value={status} onChange={(e) => setStatus(e.target.value)} className="h-8 text-xs">
            <option value="open">Open</option><option value="">All</option><option value="approved">Approved</option><option value="dismissed">Dismissed</option><option value="adjusted">Adjusted</option>
          </Select>
          <Select aria-label="Sort" value={sort} onChange={(e) => setSort(e.target.value)} className="h-8 text-xs">
            <option value="dollars">Biggest money first</option><option value="date">Newest first</option>
          </Select>
        </>} />
      {error && <InlineAlert title="Proposals could not load" onRetry={reload}>{error}</InlineAlert>}
      {loading && !data && <div className="grid gap-4 md:grid-cols-2">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-44" />)}</div>}
      {data && data.length === 0 && <EmptyState title="Nothing in this queue">Run the time machine to raise proposals, or change the filters.</EmptyState>}
      <div className="grid gap-4 xl:grid-cols-2">
        {data?.map((p) => <ProposalCard key={p.proposal_id} p={p} compact onDecided={reload} />)}
      </div>
    </>
  );
}

export function ProposalDetail() {
  const { id } = useParams();
  const { data, error, loading, reload } = useApi<any>(`/proposals/${id}`);
  return (
    <>
      <nav className="mb-3 text-xs text-muted-foreground" aria-label="Breadcrumb"><Link className="hover:underline" to="/proposals">Proposals</Link> / <span>{id}</span></nav>
      {error && <InlineAlert title="This proposal could not be loaded" onRetry={reload}>{error}</InlineAlert>}
      {loading && !data && <Skeleton className="h-96" />}
      {data && (
        <>
          <ProposalCard p={data} onDecided={reload} />
          <Card className="mt-6"><CardContent className="space-y-3 p-4">
            <h2 className="text-base font-semibold">What the agent will do if you approve</h2>
            <pre className="overflow-x-auto rounded-md bg-menu-hover p-3 text-xs">{JSON.stringify(data.action, null, 2)}</pre>
            <p className="text-xs text-muted-foreground">Nothing is sent or changed by this lab. In FoodFlow this would become a draft that a person confirms.</p>
            {data.decisions?.length > 0 && <div><h3 className="text-sm font-medium">Decision history</h3><ul className="text-xs">{data.decisions.map((d: any, i: number) => <li key={i}>{d.at}: {d.decision}{d.reason ? ` (${d.reason})` : ""} by {d.by}</li>)}</ul></div>}
          </CardContent></Card>
        </>
      )}
      <p className="mt-6 text-xs text-muted-foreground">{money(0)} is the reference zero in this lab. Figures are for method testing, not production.</p>
    </>
  );
}
