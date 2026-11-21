import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Cpu,
  Download,
  Eye,
  EyeOff,
  Gauge,
  Info,
  KeyRound,
  Plus,
  RotateCcw,
  Save,
  Settings as SettingsIcon,
  ShieldCheck,
  Trash2,
  X,
} from "lucide-react";
import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { api, type KeyProbe, type SettingsSection, type SettingsView } from "../lib/api";
import { errorMessage } from "../lib/format";
import { cn, formatBytes } from "../lib/utils";
import { Button, Card, IconButton, Spinner } from "./ui";

// Plain-language labels for every numeric setting. The server owns the bounds and defaults; this only says what each one does.
const FIELDS: Record<SettingsSection, { name: string; label: string; help: string; unit: string }[]> = {
  agent: [
    { name: "part_steps", label: "Steps per part", help: "Planning steps before Prophet pauses at a checkpoint.", unit: "steps" },
    { name: "tool_calls_per_part", label: "Tool calls per part", help: "Analyses, lookups and forecasts allowed in one part.", unit: "calls" },
    { name: "part_seconds", label: "Time per part", help: "Wall-clock time for one part. Longer work continues at the next checkpoint.", unit: "seconds" },
    { name: "sub_steps", label: "Steps for a delegated task", help: "Budget for a sub-agent that handles one self-contained piece of work.", unit: "steps" },
  ],
  context: [
    { name: "view_chars", label: "Detail kept in view", help: "Once the detail the model sees passes this size, older results fold into memory.", unit: "characters" },
    { name: "keep_recent", label: "Newest results kept verbatim", help: "The latest results always stay word for word, so the model can cite them exactly.", unit: "results" },
    { name: "memory_chars", label: "Memory size", help: "The condensed memory of older work. It is kept stable so the prompt prefix can be reused.", unit: "characters" },
  ],
};

const SECTION_TITLES: Record<SettingsSection, string> = { agent: "Agent limits", context: "Context and memory" };

function Row({ label, help, children, htmlFor }: { label: string; help?: string; children: ReactNode; htmlFor?: string }) {
  return (
    <div className="grid gap-2 border-t border-sand-2/15 py-3 first:border-t-0 first:pt-0 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-center sm:gap-6">
      <div className="grid min-w-0 gap-0.5">
        <label htmlFor={htmlFor} className="text-base font-semibold text-sand-0">
          {label}
        </label>
        {help && <span className="text-caption text-sand-1">{help}</span>}
      </div>
      <div className="min-w-0">{children}</div>
    </div>
  );
}

function Section({ id, title, description, children }: { id: string; title: string; description?: string; children: ReactNode }) {
  return (
    <section aria-labelledby={id} className="grid gap-3">
      <div className="grid gap-1">
        <h3 id={id} className="text-lg font-bold text-sand-0">
          {title}
        </h3>
        {description && <p className="text-caption text-sand-1">{description}</p>}
      </div>
      <Card className="p-4 sm:p-5">{children}</Card>
    </section>
  );
}

function Badge({ ok, yes, no }: { ok: boolean; yes: string; no: string }) {
  return (
    <span className={cn("inline-flex items-center gap-1.5 text-caption", ok ? "text-ok" : "text-ember")}>
      {ok ? <CheckCircle2 aria-hidden="true" className="size-4" /> : <AlertCircle aria-hidden="true" className="size-4" />}
      {ok ? yes : no}
    </span>
  );
}

export function SettingsModal({ open, onClose, onDataChanged }: { open: boolean; onClose: () => void; onDataChanged: () => void }) {
  const [view, setView] = useState<SettingsView | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const titleId = useId();
  const closeButton = useRef<HTMLButtonElement>(null);
  const returnTo = useRef<HTMLElement | null>(null);

  const load = () =>
    api
      .settings()
      .then((v) => {
        setView(v);
        setLoadError(null);
      })
      .catch((err) => setLoadError(errorMessage(err, "Couldn't load settings. Check that Prophet is running.")));

  useEffect(() => {
    if (!open) return;
    returnTo.current = document.activeElement as HTMLElement | null;
    void load();
    closeButton.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      returnTo.current?.focus();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 grid place-items-center p-3 sm:p-6">
      <button type="button" tabIndex={-1} aria-hidden="true" onClick={onClose} className="absolute inset-0 bg-black/60" />
      <div role="dialog" aria-modal="true" aria-labelledby={titleId} className="relative flex max-h-[92dvh] w-full max-w-3xl flex-col rounded-2xl border border-sand-2/25 bg-night-1 shadow-2xl">
        <header className="flex items-center justify-between gap-3 border-b border-sand-2/20 px-5 py-4">
          <h2 id={titleId} className="flex items-center gap-2.5 font-display text-2xl font-semibold">
            <SettingsIcon aria-hidden="true" className="size-5 text-sand-1" />
            Settings
          </h2>
          <IconButton ref={closeButton} label="Close settings" icon={<X aria-hidden="true" className="size-5" />} onClick={onClose} />
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5">
          {loadError && (
            <p role="alert" className="text-base text-bad">
              {loadError}
            </p>
          )}
          {!view && !loadError && (
            <p role="status" className="flex items-center gap-2 text-base text-sand-1">
              <Spinner /> Loading settings
            </p>
          )}
          {view && (
            <div className="grid gap-8">
              <KeysSection view={view} onSaved={setView} />
              <ModelsAndTimesFm view={view} />
              <LimitsSection view={view} onSaved={setView} />
              <SandboxSection view={view} />
              <DataSection view={view} onCleared={() => { void load(); onDataChanged(); }} />
              <AboutSection view={view} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// -- NIM keys -----------------------------------------------------------------------------------------------------------

function KeysSection({ view, onSaved }: { view: SettingsView; onSaved: (v: SettingsView) => void }) {
  const headingId = useId();
  const [keys, setKeys] = useState<string[]>(() => view.nim.keys.map((k) => k.masked));
  const [draft, setDraft] = useState("");
  const [reveal, setReveal] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [probes, setProbes] = useState<KeyProbe[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => setKeys(view.nim.keys.map((k) => k.masked)), [view.nim.keys]);

  const dirty = JSON.stringify(keys) !== JSON.stringify(view.nim.keys.map((k) => k.masked)) || draft.trim() !== "";

  function addKey() {
    const value = draft.trim();
    if (!value) return;
    if (keys.includes(value)) {
      setError("That key is already in the list.");
      return;
    }
    setKeys((k) => [...k, value]);
    setDraft("");
    setError(null);
  }

  async function save() {
    setSaving(true);
    setError(null);
    try {
      const pending = draft.trim() ? [...keys, draft.trim()] : keys;
      onSaved(await api.saveKeys(pending));
      setDraft("");
      setProbes(null);
    } catch (err) {
      setError(errorMessage(err, "Couldn't save the keys. Check the format and try again."));
    } finally {
      setSaving(false);
    }
  }

  async function test() {
    setTesting(true);
    setError(null);
    try {
      setProbes((await api.testKeys()).keys);
    } catch (err) {
      setError(errorMessage(err, "Couldn't test the keys. Save a key first."));
    } finally {
      setTesting(false);
    }
  }

  const pool = view.nim.pool;

  return (
    <Section
      id={headingId}
      title="Model keys"
      description="NVIDIA NIM keys. Prophet spreads work across them, so several keys let more work run at once. Keys are never shown in full."
    >
      <div className="grid gap-4">
        {keys.length === 0 ? (
          <p className="flex items-center gap-2 text-base text-ember">
            <AlertCircle aria-hidden="true" className="size-4" /> No key is set. Prophet cannot plan until you add one.
          </p>
        ) : (
          <ul className="grid gap-2" aria-label="Configured keys">
            {keys.map((k, i) => (
              <li key={`${i}-${k}`} className="flex items-center justify-between gap-3 rounded-lg border border-sand-2/25 bg-night-0/50 px-3 py-2">
                <span className="flex min-w-0 items-center gap-2.5">
                  <KeyRound aria-hidden="true" className="size-4 shrink-0 text-sand-2" />
                  <span className="font-mono text-meta text-sand-0 [overflow-wrap:anywhere]">
                    Key {i + 1} · {k}
                  </span>
                </span>
                <IconButton
                  tone="danger"
                  label={`Remove key ${i + 1}`}
                  icon={<Trash2 aria-hidden="true" className="size-4" />}
                  onClick={() => setKeys((list) => list.filter((_, j) => j !== i))}
                />
              </li>
            ))}
          </ul>
        )}

        <div className="grid gap-2">
          <label htmlFor={`${headingId}-new`} className="text-base font-semibold text-sand-0">
            Add a key
          </label>
          <div className="flex items-center gap-2">
            <input
              id={`${headingId}-new`}
              type={reveal ? "text" : "password"}
              value={draft}
              autoComplete="off"
              spellCheck={false}
              placeholder="nvapi-..."
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), addKey())}
              className="min-h-11 min-w-0 flex-1 rounded-lg border border-sand-2/40 bg-night-0 px-3 font-mono text-meta text-sand-0 placeholder:text-sand-2 focus:border-ember focus:outline-none"
            />
            <IconButton label={reveal ? "Hide key text" : "Show key text"} icon={reveal ? <EyeOff aria-hidden="true" className="size-5" /> : <Eye aria-hidden="true" className="size-5" />} onClick={() => setReveal((r) => !r)} />
            <IconButton label="Add key to the list" icon={<Plus aria-hidden="true" className="size-5" />} onClick={addKey} disabled={!draft.trim()} />
          </div>
        </div>

        {error && (
          <p role="alert" className="text-caption text-bad">
            {error}
          </p>
        )}

        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={() => void save()} disabled={!dirty || saving}>
            {saving ? <Spinner /> : <Save aria-hidden="true" className="size-4" />}
            Save keys
          </Button>
          <Button variant="secondary" onClick={() => void test()} disabled={testing || view.nim.keys.length === 0}>
            {testing ? <Spinner /> : <Activity aria-hidden="true" className="size-4" />}
            Test keys
          </Button>
        </div>

        {probes && (
          <ul className="grid gap-1.5 border-t border-sand-2/15 pt-4" aria-label="Key test results">
            {probes.map((p) => (
              <li key={p.slot} className="flex flex-wrap items-center gap-x-3 gap-y-1 text-caption">
                <Badge ok={p.ok} yes={`Key ${p.slot} works`} no={`Key ${p.slot} failed`} />
                <span className="font-mono text-meta text-sand-2">
                  {p.masked} · {p.status ?? "no response"} · {p.elapsed_s.toFixed(1)}s
                </span>
                {!p.ok && p.error && <span className="text-bad">{p.error}</span>}
              </li>
            ))}
          </ul>
        )}

        <div className="grid gap-2 border-t border-sand-2/15 pt-4">
          <p className="text-base font-semibold text-sand-0">Models</p>
          <ul className="flex flex-wrap gap-2" aria-label="Models in the fallback chain">
            {view.nim.models.map((m) => (
              <li key={m} className="rounded-full border border-sand-2/35 px-3 py-1 font-mono text-meta text-sand-1 [overflow-wrap:anywhere]">
                {m}
              </li>
            ))}
          </ul>
          <p className="text-caption text-sand-2">Models are tried in this order. If one is busy or unavailable, the next one takes over.</p>
        </div>

        {pool && (
          <div className="grid gap-2 border-t border-sand-2/15 pt-4">
            <p className="flex items-center gap-2 text-base font-semibold text-sand-0">
              <Gauge aria-hidden="true" className="size-4 text-sand-1" />
              Right now: {pool.in_flight} {pool.in_flight === 1 ? "request" : "requests"} running
            </p>
            <ul className="grid gap-1 text-caption text-sand-1 tabular-nums">
              {pool.keys.map((k) => (
                <li key={k.slot}>
                  Key {k.slot}: {k.in_flight} running · {k.sent_last_minute} of {k.per_minute} sent this minute
                  {k.retired ? ` · paused for ${Math.ceil(k.retired_for_s)}s after an error` : ""}
                  {k.cooling.length > 0 ? ` · cooling: ${k.cooling.map((c) => c.model).join(", ")}` : ""}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Section>
  );
}

// -- Models and TimesFM 3 -----------------------------------------------------------------------------------------------

function ModelsAndTimesFm({ view }: { view: SettingsView }) {
  const headingId = useId();
  return (
    <Section id={headingId} title="Forecast engine" description="TimesFM 3 runs locally for forecasts. Its weights have a research-only licence.">
      <dl className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,2fr)] sm:gap-x-6">
        <dt className="text-base font-semibold text-sand-0">Engine</dt>
        <dd>
          <Badge ok={view.timesfm.engine_installed} yes="Installed on this machine" no="Not installed. Forecasts fall back to the baselines." />
        </dd>
        <dt className="text-base font-semibold text-sand-0">Checkpoint</dt>
        <dd className="font-mono text-meta text-sand-1 [overflow-wrap:anywhere]">{view.timesfm.checkpoint}</dd>
      </dl>
    </Section>
  );
}

// -- Limits -------------------------------------------------------------------------------------------------------------

function LimitsSection({ view, onSaved }: { view: SettingsView; onSaved: (v: SettingsView) => void }) {
  const headingId = useId();
  const [draft, setDraft] = useState<Record<SettingsSection, Record<string, string>>>(() => toDraft(view));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => setDraft(toDraft(view)), [view]);

  const problems: string[] = [];
  const changed: Partial<Record<SettingsSection, Record<string, number>>> = {};
  for (const section of Object.keys(FIELDS) as SettingsSection[]) {
    for (const f of FIELDS[section]) {
      const [low, high] = view.bounds[section][f.name];
      const raw = draft[section][f.name];
      const n = Number(raw);
      if (raw.trim() === "" || !Number.isInteger(n) || n < low || n > high) {
        problems.push(`${f.label} must be a whole number from ${low.toLocaleString("en-AU")} to ${high.toLocaleString("en-AU")}.`);
        continue;
      }
      if (n !== view[section][f.name]) changed[section] = { ...changed[section], [f.name]: n };
    }
  }
  const dirty = Object.keys(changed).length > 0;

  async function save() {
    setSaving(true);
    setError(null);
    try {
      onSaved(await api.saveLimits(changed));
    } catch (err) {
      setError(errorMessage(err, "Couldn't save these limits."));
    } finally {
      setSaving(false);
    }
  }

  function restoreDefaults() {
    setDraft((d) => {
      const next = { ...d };
      for (const section of Object.keys(view.defaults) as SettingsSection[]) {
        next[section] = Object.fromEntries(Object.entries(view.defaults[section]).map(([k, v]) => [k, String(v)]));
      }
      return next;
    });
  }

  return (
    <Section
      id={headingId}
      title="Limits and memory"
      description="How much work Prophet does before it pauses. Pausing is never a failure: you can continue from the same place."
    >
      <div>
        {(Object.keys(FIELDS) as SettingsSection[]).map((section) => (
          <fieldset key={section} className="grid min-w-0">
            <legend className="mb-1 text-base font-bold text-sand-1">{SECTION_TITLES[section]}</legend>
            {FIELDS[section].map((f) => {
              const id = `${headingId}-${section}-${f.name}`;
              const [low, high] = view.bounds[section][f.name];
              return (
                <Row key={f.name} label={f.label} help={`${f.help} Range ${low.toLocaleString("en-AU")} to ${high.toLocaleString("en-AU")} ${f.unit}.`} htmlFor={id}>
                  <div className="flex items-center gap-2">
                    <input
                      id={id}
                      type="number"
                      inputMode="numeric"
                      min={low}
                      max={high}
                      step={1}
                      value={draft[section][f.name]}
                      onChange={(e) => setDraft((d) => ({ ...d, [section]: { ...d[section], [f.name]: e.target.value } }))}
                      className="min-h-11 w-36 rounded-lg border border-sand-2/40 bg-night-0 px-3 text-right font-mono text-base text-sand-0 tabular-nums focus:border-ember focus:outline-none"
                    />
                    <span className="w-20 text-caption text-sand-2">{f.unit}</span>
                  </div>
                </Row>
              );
            })}
          </fieldset>
        ))}
      </div>

      {problems.length > 0 && (
        <p role="alert" className="mt-3 text-caption text-bad">
          {problems[0]}
        </p>
      )}
      {error && (
        <p role="alert" className="mt-3 text-caption text-bad">
          {error}
        </p>
      )}

      <div className="mt-4 flex flex-wrap items-center justify-end gap-2 border-t border-sand-2/15 pt-4">
        <IconButton label="Restore default limits" icon={<RotateCcw aria-hidden="true" className="size-4" />} onClick={restoreDefaults} />
        <Button onClick={() => void save()} disabled={!dirty || problems.length > 0 || saving}>
          {saving ? <Spinner /> : <Save aria-hidden="true" className="size-4" />}
          Save limits
        </Button>
      </div>
    </Section>
  );
}

function toDraft(view: SettingsView): Record<SettingsSection, Record<string, string>> {
  return {
    agent: Object.fromEntries(Object.entries(view.agent).map(([k, v]) => [k, String(v)])),
    context: Object.fromEntries(Object.entries(view.context).map(([k, v]) => [k, String(v)])),
  };
}

// -- Sandbox ------------------------------------------------------------------------------------------------------------

function SandboxSection({ view }: { view: SettingsView }) {
  const headingId = useId();
  const s = view.sandbox;
  return (
    <Section id={headingId} title="Analysis sandbox" description="Where Prophet's Python analysis runs. These are fixed by the server and shown for your information.">
      <dl className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,2fr)] sm:gap-x-6">
        <dt className="text-base font-semibold text-sand-0">Time limit per run</dt>
        <dd className="text-caption text-sand-1 tabular-nums">{s.timeout_s} seconds</dd>
        <dt className="text-base font-semibold text-sand-0">Memory limit per run</dt>
        <dd className="grid gap-1 text-caption text-sand-1 tabular-nums">
          {s.mem_mb} MB
          <Badge ok={s.memory_limit_enforced} yes="Enforced by the operating system" no="Not enforced on this system" />
        </dd>
        <dt className="text-base font-semibold text-sand-0">Output size limit</dt>
        <dd className="text-caption text-sand-1 tabular-nums">{s.file_mb} MB</dd>
        <dt className="text-base font-semibold text-sand-0">Isolation</dt>
        <dd className="flex gap-2 text-caption text-sand-1">
          <ShieldCheck aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-sand-2" />
          {s.isolation}
        </dd>
      </dl>
    </Section>
  );
}

// -- Data ---------------------------------------------------------------------------------------------------------------

function DataSection({ view, onCleared }: { view: SettingsView; onCleared: () => void }) {
  const headingId = useId();
  const phrase = view.about.clear_phrase;
  const [confirm, setConfirm] = useState("");
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);
  const d = view.data;

  async function clear() {
    setClearing(true);
    setError(null);
    setDone(null);
    try {
      await api.clearData(confirm);
      setConfirm("");
      setDone("Forecasts, notebooks and uploaded files were deleted. Settings and keys were kept.");
      onCleared();
    } catch (err) {
      setError(errorMessage(err, "Couldn't delete the data. Stop any running work and try again."));
    } finally {
      setClearing(false);
    }
  }

  return (
    <Section id={headingId} title="Your data" description="Everything Prophet has stored on this machine.">
      <dl className="grid grid-cols-2 gap-3 text-caption text-sand-1 sm:grid-cols-4">
        <div className="grid gap-0.5">
          <dt className="text-sand-2">Forecasts</dt>
          <dd className="text-lg font-semibold text-sand-0 tabular-nums">{d.missions}</dd>
        </div>
        <div className="grid gap-0.5">
          <dt className="text-sand-2">Notebooks</dt>
          <dd className="text-lg font-semibold text-sand-0 tabular-nums">{d.notebooks}</dd>
        </div>
        <div className="grid gap-0.5">
          <dt className="text-sand-2">Files</dt>
          <dd className="text-lg font-semibold text-sand-0 tabular-nums">{d.datasets}</dd>
        </div>
        <div className="grid gap-0.5">
          <dt className="text-sand-2">Stored</dt>
          <dd className="text-lg font-semibold text-sand-0 tabular-nums">{formatBytes(d.bytes)}</dd>
        </div>
      </dl>

      <div className="mt-4 grid gap-4 border-t border-sand-2/15 pt-4">
        <Row label="Export everything" help="A JSON file with your forecasts, questions, steps, notebooks and settings. Keys are never included.">
          <a
            href="/api/settings/export"
            download="prophet-export.json"
            className="inline-flex min-h-11 items-center gap-2 rounded-lg border border-sand-2/40 px-4 text-base font-semibold text-sand-0 transition-colors hover:border-sand-1 hover:bg-night-2"
          >
            <Download aria-hidden="true" className="size-4" />
            Export JSON
          </a>
        </Row>

        <Row
          label="Delete all data"
          help={`Removes forecasts, notebooks and uploaded files. Settings and keys stay. Type “${phrase}” to confirm.`}
          htmlFor={`${headingId}-confirm`}
        >
          <div className="flex flex-wrap items-center gap-2">
            <input
              id={`${headingId}-confirm`}
              type="text"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              autoComplete="off"
              placeholder={phrase}
              className="min-h-11 w-full min-w-0 rounded-lg border border-sand-2/40 bg-night-0 px-3 text-base text-sand-0 placeholder:text-sand-2 focus:border-ember focus:outline-none sm:w-52"
            />
            <Button variant="secondary" onClick={() => void clear()} disabled={confirm !== phrase || clearing} className="text-bad hover:border-bad">
              {clearing ? <Spinner /> : <Trash2 aria-hidden="true" className="size-4" />}
              Delete all data
            </Button>
          </div>
        </Row>

        {error && (
          <p role="alert" className="text-caption text-bad">
            {error}
          </p>
        )}
        {done && (
          <p role="status" className="text-caption text-ok">
            {done}
          </p>
        )}
      </div>
    </Section>
  );
}

// -- About --------------------------------------------------------------------------------------------------------------

// The licence notice lives here and nowhere on the working surfaces: it is attribution, not a banner.
function AboutSection({ view }: { view: SettingsView }) {
  const headingId = useId();
  return (
    <Section id={headingId} title="About">
      <div className="grid gap-3">
        <p className="flex items-center gap-2 text-base font-semibold text-sand-0">
          <Cpu aria-hidden="true" className="size-4 text-sand-1" />
          {view.about.name}
        </p>
        <p className="flex gap-2 text-caption text-sand-1">
          <Info aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-sand-2" />
          {view.about.licence}
        </p>
      </div>
    </Section>
  );
}
