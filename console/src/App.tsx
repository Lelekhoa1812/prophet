import { NavLink, Route, Routes, Navigate } from "react-router-dom";
import { Moon, Sun, Eye, EyeOff, Compass, Briefcase, Code2, Presentation } from "lucide-react";
import { useApp, type Lens } from "@/lib/context";
import { Button, InlineAlert, Select } from "@/components/ui";
import { cn } from "@/lib/utils";
import { Overview } from "@/pages/Overview";
import { Briefing } from "@/pages/Briefing";
import { Proposals, ProposalDetail } from "@/pages/Proposals";
import { Agents } from "@/pages/Agents";
import { Explorer } from "@/pages/Explorer";
import { Scorecard } from "@/pages/Scorecard";
import { TimeMachine } from "@/pages/TimeMachine";
import { Developer } from "@/pages/Developer";
import { Method } from "@/pages/Method";

const LENSES: { key: Lens; label: string; icon: typeof Compass; hint: string }[] = [
  { key: "executive", label: "Executive", icon: Presentation, hint: "Headline, dollars and trust" },
  { key: "operator", label: "Operator", icon: Briefcase, hint: "What needs you today" },
  { key: "developer", label: "Developer", icon: Code2, hint: "Models, runs and API" },
  { key: "demo", label: "Client demo", icon: Compass, hint: "Scripted walkthrough" },
];

const NAV: Record<Lens, { to: string; label: string }[]> = {
  executive: [{ to: "/", label: "Overview" }, { to: "/scorecard", label: "Proof" }, { to: "/method", label: "How it works" }, { to: "/time", label: "Time machine" }],
  operator: [{ to: "/briefing", label: "Briefing" }, { to: "/proposals", label: "Proposals" }, { to: "/agents", label: "Agents" }, { to: "/time", label: "Time machine" }],
  developer: [{ to: "/developer", label: "Models and runs" }, { to: "/explorer", label: "Forecast explorer" }, { to: "/agents", label: "Agents and triggers" }, { to: "/scorecard", label: "Scorecard" }, { to: "/proposals", label: "Proposals" }, { to: "/method", label: "Method" }, { to: "/time", label: "Time machine" }],
  demo: [{ to: "/", label: "Start" }, { to: "/briefing", label: "Morning briefing" }, { to: "/explorer", label: "Forecast" }, { to: "/scorecard", label: "Proof" }],
};

export default function App() {
  const { meta, metaError, dataset, setDataset, lens, setLens, dark, toggleDark, reveal, setReveal } = useApp();
  const ds = meta?.datasets ?? {};
  const available = Object.keys(ds);
  const defaultPath = { executive: "/", operator: "/briefing", developer: "/developer", demo: "/" }[lens];
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 border-b bg-background/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-2.5 lg:px-6">
          <div className="flex items-center gap-2.5">
            <span className="grid h-8 w-8 place-items-center rounded-md bg-primary text-primary-foreground font-semibold">T</span>
            <div className="leading-tight">
              <p className="text-sm font-semibold">TimesFM ERP Lab</p>
              <p className="text-[11px] text-muted-foreground">Forecasting and anomaly agents for food distribution</p>
            </div>
          </div>
          <nav aria-label="Lens" className="flex flex-wrap gap-1 rounded-lg bg-[#e7ecf2] p-1 dark:bg-slate-800">
            {LENSES.map((l) => (
              <button key={l.key} onClick={() => setLens(l.key)} title={l.hint}
                className={cn("inline-flex h-7 items-center gap-1.5 rounded-md px-2.5 text-xs font-medium cursor-pointer transition-colors",
                  lens === l.key ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground")}
                aria-pressed={lens === l.key}>
                <l.icon className="h-3.5 w-3.5" aria-hidden />{l.label}
              </button>
            ))}
          </nav>
          <div className="ml-auto flex flex-wrap items-center gap-2">
            <label className="flex items-center gap-2 text-xs text-muted-foreground">Dataset
              <Select aria-label="Dataset" value={dataset} onChange={(e) => setDataset(e.target.value)} className="h-8 text-xs">
                {available.length === 0 && <option value={dataset}>{dataset}</option>}
                {available.map((k) => <option key={k} value={k}>{k === "twin" ? "Synthetic twin (labelled)" : "Staging (real, 8 weeks)"}</option>)}
              </Select>
            </label>
            <Button variant="outline" size="sm" onClick={() => setReveal(!reveal)} aria-pressed={reveal} title="Show real names. Local only; the reveal map never leaves this machine.">
              {reveal ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />}{reveal ? "Names shown" : "Names hidden"}
            </Button>
            <Button variant="ghost" size="icon" onClick={toggleDark} aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}>
              {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
          </div>
        </div>
        <div className="mx-auto flex max-w-[1400px] gap-1 overflow-x-auto px-4 pb-2 lg:px-6" aria-label="Pages">
          {NAV[lens].map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === "/"}
              className={({ isActive }) => cn("whitespace-nowrap rounded-md px-3 py-1.5 text-sm cursor-pointer", isActive ? "bg-accent font-semibold text-accent-foreground" : "text-muted-foreground hover:bg-menu-hover hover:text-foreground")}>
              {n.label}
            </NavLink>
          ))}
        </div>
      </header>
      <main className="mx-auto max-w-[1400px] px-4 py-6 lg:px-6">
        {metaError && <div className="mb-4"><InlineAlert title="The lab API is not reachable">Start it with <code className="font-mono">uvicorn lab.api.main:app --port 8000</code> from the project folder. ({metaError})</InlineAlert></div>}
        {meta && !meta.model.loaded && <p className="mb-4 text-xs text-muted-foreground">The TimesFM model loads on first forecast (about 80 seconds). Pages that only read stored results are unaffected.</p>}
        {meta && <LicenceBanner licence={meta.model.licence} />}
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/briefing" element={<Briefing />} />
          <Route path="/proposals" element={<Proposals />} />
          <Route path="/proposals/:id" element={<ProposalDetail />} />
          <Route path="/agents" element={<Agents />} />
          <Route path="/explorer" element={<Explorer />} />
          <Route path="/scorecard" element={<Scorecard />} />
          <Route path="/time" element={<TimeMachine />} />
          <Route path="/developer" element={<Developer />} />
          <Route path="/method" element={<Method />} />
          <Route path="*" element={<Navigate to={defaultPath} replace />} />
        </Routes>
      </main>
      <footer className="mx-auto max-w-[1400px] px-4 pb-8 pt-4 text-xs text-muted-foreground lg:px-6">
        Lab only. Read-only against FoodFlow; nothing here writes to a client system. Synthetic twin results are labelled as such. Figures are demonstrations of method, not forecasts.
      </footer>
    </div>
  );
}

function LicenceBanner({ licence }: { licence: string }) {
  return (
    <div className="mb-5 flex items-start gap-3 rounded-lg border border-amber-300 bg-amber-50 p-3 text-xs text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100">
      <span className="font-semibold">Licence gate.</span><span>{licence} This lab is R&amp;D only.</span>
    </div>
  );
}
