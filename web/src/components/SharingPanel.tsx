import { useEffect, useState } from "react";
import { api, type NotebookSharing } from "../lib/api";
import { errorMessage } from "../lib/format";
import { Button } from "./ui";

// What this notebook may share (A7-12): web research on or off, and which model providers may receive its data.
// The summary is always visible so a person can see what is enabled; the controls open on demand. Changes apply to
// questions asked after the change, and to automatic parts that start after it.
export function SharingPanel({ notebookId }: { notebookId: string }) {
  const [sharing, setSharing] = useState<NotebookSharing | null>(null);
  const [open, setOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setSharing(null);
    setError(null);
    api
      .notebookSharing(notebookId)
      .then((s) => {
        if (live) setSharing(s);
      })
      .catch((err) => {
        if (live) setError(errorMessage(err, "Sharing settings could not be loaded."));
      });
    return () => {
      live = false;
    };
  }, [notebookId]);

  if (error && !sharing) {
    return (
      <p role="alert" className="text-caption text-bad">
        {error}
      </p>
    );
  }
  if (!sharing) return null;

  const allowed = sharing.providers ?? sharing.available.map((p) => p.name);
  const allowedLabels = sharing.available.filter((p) => allowed.includes(p.name)).map((p) => p.label);
  const providerText =
    sharing.providers === null
      ? "every configured model provider may receive its data"
      : allowedLabels.length > 0
        ? `only ${allowedLabels.join(", ")} may receive its data`
        : "no model provider may receive its data";
  const summary = `${sharing.web_research ? "web research is on" : "web research is off"}; ${providerText}.`;

  async function save(patch: { web_research?: boolean; providers?: string[] | null }) {
    setSaving(true);
    setError(null);
    try {
      setSharing(await api.setNotebookSharing(notebookId, patch));
    } catch (err) {
      setError(errorMessage(err, "Sharing settings could not be saved. Try again."));
    } finally {
      setSaving(false);
    }
  }

  function toggleProvider(name: string, on: boolean) {
    if (!sharing) return;
    const next = on ? [...allowed, name] : allowed.filter((n) => n !== name);
    const everyone = sharing.available.every((p) => next.includes(p.name));
    void save({ providers: everyone ? null : next });
  }

  return (
    <section aria-label="Data sharing for this notebook" className="mb-3 rounded-xl border border-sand-2/30 p-3 text-caption">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="min-w-0 text-sand-1">
          <span className="font-semibold text-sand-0">Sharing:</span> {summary}
        </p>
        <Button variant="ghost" size="sm" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
          {open ? "Close" : "Change sharing"}
        </Button>
      </div>

      {error && (
        <p role="alert" className="mt-2 text-bad">
          {error}
        </p>
      )}

      {open && (
        <fieldset disabled={saving} className="mt-3 grid gap-4">
          <legend className="sr-only">Data sharing for this notebook</legend>

          <label className="flex items-start gap-3">
            <input
              type="checkbox"
              className="mt-0.5 size-4 accent-ember"
              checked={sharing.web_research}
              onChange={(e) => void save({ web_research: e.target.checked })}
            />
            <span className="grid gap-0.5">
              <span className="text-sand-0">Web research</span>
              <span className="text-sand-1">
                {sharing.web_research
                  ? "Prophet may search and read public web pages for questions in this notebook."
                  : "Prophet will not search or read web pages for questions in this notebook."}
              </span>
            </span>
          </label>

          <div className="grid gap-2">
            <p className="text-sand-0">Model providers that may receive this notebook&apos;s data</p>
            <ul className="grid gap-2">
              {sharing.available.map((p) => {
                const checked = allowed.includes(p.name);
                const status = p.ready ? "ready" : p.enabled ? "no key yet" : "turned off in Settings";
                return (
                  <li key={p.name}>
                    <label className="flex items-center gap-3">
                      <input
                        type="checkbox"
                        className="size-4 accent-ember"
                        checked={checked}
                        onChange={(e) => toggleProvider(p.name, e.target.checked)}
                      />
                      <span className="text-sand-0">{p.label}</span>
                      <span className="text-sand-2">({status})</span>
                    </label>
                  </li>
                );
              })}
            </ul>
            <p className="text-sand-2">Unticking a provider stops this notebook sending its data to that provider.</p>
          </div>
        </fieldset>
      )}
    </section>
  );
}
