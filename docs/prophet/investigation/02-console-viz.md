# 02. Console visualisation and UI inventory (for the Prophet rewrite)

Date: 20 Nov 2026. Scope: `console/` (React 18, Vite 6, Tailwind 4) and the route list in `src/lab/api/main.py`.
Method: static reading of every file in `console/src/` plus config, `docs/DESIGN_REVIEW.md`, and installed packages read from `console/node_modules` (read-only). No existing file was edited, no `npm install`, no build, no dev server. One Node check was run with `TZ=Australia/Sydney` to confirm a date bug (section 6). Citations are `file:line` relative to `/Users/liamle/Downloads/prophet`.

Headline: the chart components (`BandChart.tsx`, `Bars.tsx`) and the UI primitives (`components/ui/index.tsx`) are the reusable core. The pages, the proposal card and the copy are ERP-specific. There is no chat, message or streaming component anywhere in the console.

---

## 1. Verified stack and dependencies

**Environment facts**
- `console/node_modules` exists (69 top-level entries). Installed versions below were read from `node_modules/<pkg>/package.json`.
- `console/dist` exists (13 files in `dist/assets`) and is not tracked in git (`git ls-files console` returns no `dist` entries).
- No `components.json` (no shadcn CLI config). No ESLint or Prettier config. No test runner or test script: `package.json:7-10` defines only `dev`, `build`, `typecheck`, `preview`.
- Build gate: `"build": "tsc --noEmit && vite build"` (`package.json:8`). TypeScript is `strict` (`tsconfig.json:4`), `jsx: react-jsx`, `paths: @/* -> src/*` (`tsconfig.json:3-5`). The same alias is in `vite.config.ts:9`.
- Dev proxy: only `/api` is proxied to `127.0.0.1:8000` (`vite.config.ts:10`).

**Dependency table** (declared in `package.json:13-34`)

| Package | Declared | Installed | Used at | Verdict |
|---|---|---|---|---|
| react, react-dom | ^18.3.1 | 18.3.1 | `main.tsx:8` | KEEP |
| react-router-dom | ^6.30.1 | 6.30.6 | `main.tsx:3`, `App.tsx:1`, `ProposalCard.tsx:2` | KEEP (routes will change) |
| @radix-ui/react-dialog | ^1.1.14 | 1.3.1 | `ui/index.tsx:4`, used only by `Sheet` (`ui/index.tsx:92-110`) | REMOVE: `Sheet` and `SheetContent` have zero call sites |
| @radix-ui/react-slot | ^1.2.3 | 1.5.0 | `ui/index.tsx:5,29` (Button `asChild`) | ADAPT: no caller passes `asChild`; keep only if shadcn Button compatibility is wanted |
| @radix-ui/react-switch | ^1.2.5 | 1.4.1 | `ui/index.tsx:77-81`, `pages/Agents.tsx:58` | KEEP |
| @radix-ui/react-tabs | ^1.1.12 | 1.2.1 | `ui/index.tsx:83-90`, `pages/Explorer.tsx:65-67` | KEEP |
| class-variance-authority | ^0.7.1 | 0.7.1 | `ui/index.tsx:11-26, 40-53` | KEEP |
| clsx | ^2.1.1 | 2.1.1 | `lib/utils.ts:1,4` | KEEP |
| tailwind-merge | ^3.3.0 | 3.7.0 | `lib/utils.ts:2,4` | KEEP |
| lucide-react | ^0.511.0 | 0.511.0 | 16 icons across `App.tsx`, `Flow.tsx`, `ui/index.tsx`, `ProposalCard.tsx` | KEEP |
| @fontsource-variable/montserrat | ^5.2.5 | 5.3.0 | `index.css:2` | REPLACE (brand font, see section 4) |
| @fontsource-variable/geist-mono | ^5.2.5 | 5.3.0 | `index.css:3` | KEEP or swap for brand mono |
| tailwindcss, @tailwindcss/vite | ^4.1.8 | 4.3.5 | `index.css:1`, `index.css:87-114` (CSS-first, no `tailwind.config`), `vite.config.ts:3,8` | KEEP |
| vite | ^6.3.5 | 6.4.6 | `vite.config.ts` | KEEP |
| @vitejs/plugin-react | ^4.5.2 | 4.7.0 | `vite.config.ts:2,8` | KEEP |
| typescript | ~5.8.3 | 5.8.3 | `tsconfig.json` | KEEP |
| @types/react, @types/react-dom | ^18.3.23, ^18.3.7 | 18.3.31, 18.3.7 | type-only | KEEP |

**Unused or redundant**
- Dead code, not dependencies: `Sheet`/`SheetContent` (`ui/index.tsx:92-110`) and `Sparkline` (`charts/Bars.tsx:40-45`). `Sheet` is the only reason `@radix-ui/react-dialog` is installed.
- Absent by design, so a rewrite starts from nothing: no chart library (charts are hand-built SVG in `BandChart.tsx` and `Bars.tsx`), no animation library, no React Bits, no state library (React context only, `context.tsx`), no data-fetch library (`useApi`, `api.ts:46-61`), no test runner, no lint.
- Nothing in the console uses the `--fresh`, `--ageing`, `--urgent`, `--expired` tokens (defined at `index.css:32-35`, no reference outside the CSS file).

---

## 2. Page-by-page and component-by-component inventory

ERP-specific: yes means the file's purpose or content only makes sense for FoodFlow or AXIS (stock, lots, purchase orders, customers, buyers). "Partly" means the structure is generic but the content or labels are not.

### 2a. Pages (`src/pages/`)

| File (lines) | Purpose | ERP-specific? | Verdict | Rationale |
|---|---|---|---|---|
| `Overview.tsx` (70) | Executive headline: four stat tiles, per-agent recall bars, "what this rests on" panel | Yes | ADAPT | Stat-tile grid is reusable. Ship-bar metrics (29-34) and the "meat wholesaler" copy (57) are specific. |
| `Briefing.tsx` (73) | Owner-filtered "needs you" feed, digest, "still open" rail | Yes | ADAPT | Feed plus digest is a generic inbox pattern. Owner roles are hard-coded (22: buyer, sales, account_manager, order_desk, pricing). |
| `Proposals.tsx` (9-45) and `ProposalDetail` (47-69) | Review queue with agent, status and sort filters; detail page | Partly | ADAPT | Closest analogue to a Prophet approvals inbox. Detail shows raw `action` JSON (60) and says "In FoodFlow this would become a draft" (61). |
| `Agents.tsx` (126) | Per-agent config form: enable, owner, quiet hours, cooldown, autonomy radiogroup, advanced thresholds, lessons | Yes | ADAPT | Autonomy radiogroup (82-89) and config form are generic. Trigger copy (9-14), `POLICY_AGENTS` (18) and the threshold note (108) are ERP and out of date with DESIGN_REVIEW. |
| `Explorer.tsx` (195) | Series picker, weekly/daily toggle, history and horizon controls, `BandChart`, comparison table, `ModelSaw` explainer | Mostly generic | KEEP core, ADAPT labels | Best base for Prophet's forecast view. Rename "product", "category", "shelf life" (50-56) and "Price and stock are not part of the forecast" (136). The table at 105-115 is the accessible fallback for the chart. |
| `Scorecard.tsx` (95) | Proof page: WAPE bars, calibration diagram, ship-bar tiles | Partly | ADAPT | `HBars` and `Calibration` usage is reusable. `BAR` constants (7), the "Victorian calendar" label (90) and ship-bar copy (85) are specific. |
| `TimeMachine.tsx` (65) | Step a replay day by day; results accumulate sorted by dollars | No (concept) | ADAPT | Generic replay pattern. Contains the date bug at 20-22 (section 6). |
| `Developer.tsx` (126) | Model status tiles, `Rollout`, `WhatIf`, backtest runner, jobs list, API table | Partly | ADAPT | Split it. Jobs and API table are generic. Staging role-audit warning (28-34), Bedrock spend (38) and the API table content (103-126) are specific. |
| `Method.tsx` (89) | Static four-flow explainer with a seafood worked example | Yes | REMOVE | Content is ERP-specific (22-23 "Seafood P-0120", 54, 76 "AWS Sydney"). Keep the `Flow.tsx` layout only. |

### 2b. Components (`src/components/`)

| File (lines) | Purpose | ERP-specific? | Verdict | Rationale |
|---|---|---|---|---|
| `ui/index.tsx` (153) | Primitives: `Button` (cva, 11-36), `Badge` (40-56), `Card*` (58-64), `Skeleton` (66), `Input`, `Select`, `Switch`, `Tabs*`, `InlineAlert` (112-121), `PageHeader` (123-134), `Stat` (136-145), `EmptyState`, `SEVERITY` (151-153) | No, except palette | KEEP | Primary reusable asset, shadcn-shaped. Replace hard-coded palettes (43-48, 113) and `#e7ecf2` (85). Drop `Sheet` (92-110) unless a drawer is needed. |
| `charts/BandChart.tsx` (103) | Quantile fan chart with time axis, hover readout, event shading, peak clipping | Core is generic. Legend text "TimesFM median" (81) and "Labelled event (twin)" (83) are specific. | KEEP (top reuse candidate) | Section 3.1. |
| `charts/Bars.tsx` (45) | `HBars` (4-21), `Calibration` reliability diagram (24-38), `Sparkline` (40-45, unused) | No | KEEP `HBars` and `Calibration`; REMOVE `Sparkline` | Generic. `Sparkline` has no call site. |
| `Flow.tsx` (71) | Responsive flow diagram: `FlowNode` (9-25), `FlowArrow` (27-35), `FlowRow` (38-50), `FlowMerge` (53-61), `Example` (64-71) | No (the `Example` default title "A real example from the twin", 64, is specific) | KEEP | Pure layout in HTML and CSS, so text cannot clip. Good for explaining a pipeline in Prophet. |
| `ProposalCard.tsx` (150) | One recommendation: severity, agent badge, autonomy badge, dollars at stake, AI narration, `BandChart`, facts, targets, Approve, Adjust, Dismiss menu, Open | Yes | ADAPT | Strongest candidate to become Prophet's recommendation message card in chat. Specific: `AGENT_LABEL` (10), entity code regex `^(C\|P)-\d{4}$` (22, 25), "Who usually buys this" (96-108), dismiss reasons (132), "Write it with AI (Haiku)" (72). |
| `Lessons.tsx` (52) | List of learned suppression rules with approve (two-person) and undo | Partly | ADAPT | "Learning is visible and reversible" is a generic and useful pattern. Copy and whole-agent scope are ERP. Comment at 5 cites a FoodFlow rule. |
| `Rollout.tsx` (143) | Client economics form (review cost, effectiveness, false-alarm cost, budget per owner, go-live), alert-budget frontier table, readiness check per agent, drift table, learning log | Yes | ADAPT | Frontier table, drift table and "learning log" are reusable evaluation UI. Vocabulary (stockouts, expiry, owners) is specific. |
| `WhatIf.tsx` (57) | Expected-value table with an acceptance selector | Partly | ADAPT | Generic value-estimate pattern. Column names and "claims" wording are specific. Copy at 16 ("no client has acted") is a good honesty pattern to keep. |
| `lib/api.ts` (61) | Typed `api()` wrapper (34-42), `post`/`put` (43-44), `useApi` hook (46-61), domain types (4-32) | Types yes; wrapper no | ADAPT | Keep the wrapper and hook. Replace `Proposal`, `SeriesEvidence`, `AgentInfo`, `Meta` (4-32). |
| `lib/context.tsx` (39) | Global state: meta fetch, dataset, lens, dark mode, reveal toggle, revealed-name map | Yes | ADAPT | See section 6, risk 1. |
| `lib/utils.ts` (17) | `cn`, `money`, `pct`, `num`, `dateLabel`, `dayShort` | Partly | KEEP | `money` hard-codes "$" and `en-AU` (9-11). Parameterise currency and locale. |
| `App.tsx` (118) | Shell: skip link (43), header with mark and title (47-50), lens switcher (53-62), dataset select (64-68), reveal and dark toggles, per-lens nav (79-84), routes (91-103), footer (105-107), `LicenceBanner` (112-118) | Yes | ADAPT | Shell pattern is reusable, including the skip link and the licence-banner pattern. Labels, lens definitions (16-28), dataset labels (67) and footer copy (106) are specific. |
| `main.tsx` (16) | Root render: `StrictMode`, `BrowserRouter`, `AppProvider` | No | KEEP | |
| `index.html` (13) | Title "TimesFM ERP Lab" (6); green "T" favicon (7) | Yes | REPLACE | Brand identity. |
| `index.css` (132) | Tokens, dark mode, Tailwind theme mapping, skeleton shimmer, reduced motion | Yes (brand values) | ADAPT | See section 4. Mechanism is good; values and names are FoodFlow. |
| `vite.config.ts` (11) | React and Tailwind plugins, `@` alias, `/api` proxy | No | KEEP | |
| `tsconfig.json` (8) | Strict TS, `react-jsx`, `@/*` path | No | KEEP | |
| `package.json` (2) | Name `timesfm-erp-console` | Name only | ADAPT | Rename. |

No component in the console handles chat messages, a message list, a composer, streaming, or tool-call display. Those are net-new.

---

## 3. Visualisation strategies worth preserving

### 3.1 Quantile fan chart (`charts/BandChart.tsx`)

Props (`BandChart.tsx:4-12`): `history {x, y}`, optional `forecast {x, q10, q50, q90, q20?, q80?}`, `baselines[]`, `actual {x, y: (number|null)[]}`, `truth {x, y}`, `events[{start, end, label, kind}]`, `unit`. Plus `height`, `label`, `clip` (default false).

1. **Time axis, not index axis.** Each ISO date is parsed as local midnight, `new Date(iso + "T00:00:00").getTime()` (21). The x-domain is min and max over history, forecast and truth dates (23-24). A zero-width domain is padded by 1 ms (33). Gaps in dates therefore render at their true spacing.
2. **Y scale from zero.** Top `y1 = peak * 1.08`, where peak is the max of history, q90, q10, truth and actuals (25-27). The Y map clamps values above `y1` to the top edge (36).
3. **Clip mode** (`clip=true`, used by `ProposalCard.tsx:81`). `y1 = max(p97 of all values, max q90, 1) * 1.12` (28-32). Taller points are capped and the real peak is printed above the chart (51). Purpose, from the comment at 16-17: one lumpy order cannot flatten the forecast.
4. **Geometry.** Fixed `viewBox` of 760 by `height`, margins left 48, right 14, top 12, bottom 26 (19). The SVG is `width: 100%` and scales with the viewBox. There is no resize observer.
5. **Layers, bottom to top** (order matters):
   - Event shading: `rect` across the plot height, fill `--chart-truth` at 13% opacity, minimum 3 units wide, with a `<title>` (66-69).
   - Dashed vertical line at the forecast origin (70).
   - 10 to 90 band as one closed path: q90 left to right, then q10 right to left (46-48), fill `--chart-band` (71).
   - Baselines: dashed, 1.25 px, `--chart-baseline` (72).
   - Truth: dotted, 1.5 px, `--chart-truth` (73).
   - History: solid, 1.75 px, `--chart-history` (74).
   - Forecast median: 2.25 px, `--chart-forecast` (75).
   - Actuals after the forecast: circles of radius 3.5 with a background-coloured stroke (76).
6. **Path builder with real gaps.** `line(xs, ys)` (37) emits `M` for the first point and after any null, `L` otherwise. Missing values produce breaks, not interpolation.
7. **Gridlines.** Five levels at 0, 0.25, 0.5, 0.75, 1 times `y1` (38, 62-65). Labels use `toLocaleString("en-AU")` with one decimal below 10 (45).
8. **Hover scrub.** `onMouseMove` converts `clientX` to viewBox units with `getBoundingClientRect` (54-57), inverts to a time value (57), and picks the nearest date from the union of history and forecast dates by linear scan (58-60). Then:
   - A vertical cursor line is drawn (78).
   - A readout panel (86-94, `aria-live="polite"`) shows Actual, Median with the q10 to q90 range, Actually (actuals), and True (truth). Missing fields are omitted.
   - `onMouseLeave` clears the hover (53).
9. **Legend** (80-85) built from `Key` (98-103): solid, dashed, bold and box swatches. Colour is not the only cue. Truth (dotted orange) and baseline (dashed grey) differ by dash pattern.

**Re-implementation recipe for any time series**
1. Inputs: a list of `{t, value}` for history, optional forecast `{t, p10, p50, p90}`, optional overlays (baselines, actuals, a reference truth), optional annotated spans.
2. Scales: linear x on time (use UTC or a date library, not local-midnight strings); linear y from zero with headroom. Optionally a percentile-capped y for outlier-heavy data, with a printed peak.
3. Emit SVG paths in this order: spans, origin line, band polygon, dashed overlays, history, median, point overlays.
4. Null-aware path builder that breaks lines at gaps.
5. Hover: map pointer to the nearest timestamp, show a cursor and a text readout with all series at that time.
6. Legend with dash and box swatches.

**Limitations to fix in the rewrite**
- Only the 10 to 90 band is drawn. `q20` and `q80` are typed (6) but never rendered.
- Hover is mouse-only. No keyboard focus, no touch-specific handling.
- The SVG has `role="img"` and no accessible name (52). The figure has an `aria-label` (50), which is not the same thing. The proposal-card chart (`ProposalCard.tsx:81`) has no data-table alternative. `Explorer.tsx:105-115` does.
- Nearest-date search and `indexOf` run on every mouse move and render (39-44). Fine at about 400 weekly points. Untested at the 2,000-day "All" preset (`Explorer.tsx:12`).
- Date tick labels use `toISOString` (77), which shifts them a day earlier east of UTC (section 6).

### 3.2 Comparison bars (`charts/Bars.tsx:4-21`, `HBars`)
- Grid row: label (`minmax(9rem,38%)`), track, tabular value (11).
- `better: "lower" | "higher"` decides which value is the best. Best bar uses `bg-primary`. `highlight` label uses `bg-teal`. Others use slate (14). This keeps colour and ranking honest.
- Width is `max(2%, value / max)` (14). Track has `role="img"` and an `aria-label` with the formatted value (13).
- Used for WAPE (`Scorecard.tsx:40`), agent recall (`Overview.tsx:42`), units error (`Explorer.tsx:122`).

### 3.3 Reliability diagram (`charts/Bars.tsx:24-38`, `Calibration`)
- X: nominal quantile levels 0.1 to 0.9. Y: share of actuals at or below each level. Dashed diagonal is perfect calibration (30). Points carry `<title>` tooltips (32). Ticks at 0, 50, 100 percent. Axis labels are "What the forecast said" and "What actually happened".
- Reusable for any probabilistic forecast. Note that `observed` is a plain array in quantile order (`Scorecard.tsx:49`).

### 3.4 Flow diagrams (`Flow.tsx`)
- HTML and CSS, not SVG. `FlowRow` stacks vertically below `md` and uses `md:contents` so arrows sit between nodes (38-50). `FlowMerge` feeds several inputs into one node (53-61). Text wraps instead of clipping.

### 3.5 Other reusable patterns
- **Data loading** (`api.ts:46-61`): `useApi(path | null)` skips when `null`, re-fetches when the path or a `reload` tick changes, and ignores stale responses with a `live` flag. No cache, no abort.
- **Compact money** (`utils.ts:6-12`): `$1.2M`, `$45k`, otherwise `$1,234`. Good for dashboards.
- **Stat tiles** (`ui/index.tsx:136-145`): eyebrow label, large tabular value, optional tone (good, warn, bad) mapped to `--status-*` tokens.
- **Dismiss with reason, undo** (`ProposalCard.tsx:129-150`, `Lessons.tsx`): a feedback pattern that maps directly to chat feedback on a recommendation.
- **Licence banner** (`App.tsx:112-118`): a persistent notice for a third-party model licence. Keep the pattern for any model-licence constraint.

---

## 4. Design tokens and branding to replace

Tokens are defined in `index.css`. Its own comment says they were copied from FoodFlow's `apps/web/app/globals.css` (`index.css:8-11`). The mechanism (CSS variables on `:root` and `.dark`, mapped to Tailwind through `@theme inline`) is worth keeping. The values and names are FoodFlow.

| Item | Where | What it is | Action |
|---|---|---|---|
| Brand palette | `index.css:13-16` | `--brand-navy #132a3b`, `--brand-teal #0d706d`, `--brand-green #28bf60`, `--brand-orange #f0981a` | REPLACE with Prophet palette |
| Action tokens | `index.css:17-22` | `--primary #1b7b54`, `--cta #28bf60`, `--accent #e9f5f4`, `--ring` | REPLACE values; keep names |
| Status tokens | `index.css:25-31` | `--status-success/warning/error/info/neutral`, text-safe for WCAG AA (comment 25) | KEEP the contrast approach; retune values |
| Expiry-domain tokens | `index.css:32-35` | `--fresh`, `--ageing`, `--urgent`, `--expired` (food shelf-life states) | REMOVE: no references in components |
| Chart tokens | `index.css:49-56`, dark at 79-83 | `--chart-history`, `--chart-forecast` (teal), `--chart-band`, `--chart-truth` (orange, also used for event shading), `--chart-baseline`, `--chart-grid` | Rename to semantic roles (for example `series-primary`, `annotation`); keep the mechanism |
| Dark mode | `index.css:6`, `60-85`; `context.tsx:25` | Class-based `.dark` via `@custom-variant`; toggled on `<html>` | KEEP; retune values |
| Fonts | `index.css:2-3`, `36-37` | Montserrat Variable (sans), Geist Mono (mono) | REPLACE sans per Prophet brand |
| Hard-coded colours | `App.tsx:53`, `ui/index.tsx:85` (`#e7ecf2`); Tailwind palettes in `ui/index.tsx:43-48, 113`, `Flow.tsx:14`, `App.tsx:114`, `Agents.tsx`, `Explorer.tsx`, `Developer.tsx` | Literal values outside the token system | Move into tokens |
| Identity | `index.html:6-7`; `App.tsx:47` (green "T" mark), `App.tsx:49-50` (title, subtitle "Forecasting and anomaly agents for food distribution") | Name and mark | REPLACE |
| FoodFlow rule comments | `index.css:9-10, 128`; `ui/index.tsx:10, 92`; `Lessons.tsx:5`; `Proposals.tsx:61`; `Method.tsx:54` | Comments and copy citing FoodFlow | REMOVE or rewrite |
| Footer | `App.tsx:105-107` | "Read-only against FoodFlow; nothing here writes to a client system" | REPLACE |
| Domain vocabulary | `Briefing.tsx:22` (owner roles); `ProposalCard.tsx:132` (dismiss reasons); `ProposalCard.tsx:22, 25` (entity code regex `C|P-dddd`); `Agents.tsx:9-14` (triggers: purchase orders, use-by dates); `Scorecard.tsx:90` ("Victorian calendar") | Business nouns | REPLACE with a configurable vocabulary |
| Currency and locale | `utils.ts:9-14`; `Developer.tsx:38` | Hard-coded "$" and `en-AU` | Parameterise |
| Dataset identity | `context.tsx:20` (default "twin"); `App.tsx:67` ("Synthetic twin", "Staging (real)"); `Developer.tsx:28-34` | Test-data framing | REMOVE for Prophet; replace with tenant or workspace |
| Ship-bar and proof constants | `Scorecard.tsx:7, 85`; `Overview.tsx:29-34`; `Developer.tsx:61` | Hard-coded targets (15 percent, 70 percent) | Move to config |
| Licence copy | `App.tsx:112-118`; `Overview.tsx:59`; `Developer.tsx:39` | TimesFM licence | Keep the banner pattern; swap text per model |
| Lens and nav | `App.tsx:16-28, 35` | Executive, Operator, Developer, Client demo lenses with page lists | REPLACE: the lens idea is reusable, the content is lab-specific |

Visual identity today: flat cards with a light shadow, no imagery, no gradients, no motion beyond a skeleton shimmer (`index.css:128-130`, with reduced-motion handling at 132). Nothing on the page supports a "Dune-meets-space" look, so that would be new work, and any React Bits component is a new dependency.

---

## 5. API surface

Source: `@app.*` decorators in `src/lab/api/main.py`. There are 33 `/api` routes, plus a static `/assets` mount and an SPA fallback (around lines 889-898). Classes: **C** chart or forecast data; **G** generic platform, reusable as is; **E** ERP-specific; **M** mixed.

The Developer page's API table (`Developer.tsx:103-126`) documents 22 of the 33. It omits: `GET /api/proposals/{id}`, `POST /api/lessons/{id}/approve`, `GET /api/backtests`, `GET /api/jobs`, `GET /api/products`, `GET /api/truth/events`, `POST /api/tenant/calibrate`, `POST /api/time/reset`, `GET /api/time/state`, `GET /api/policy/whatif`, `GET /api/reveal`.

| # | Method and path | main.py line | Purpose (from code or docstring) | Class | Console caller |
|---|---|---|---|---|---|
| 1 | GET /api/meta | 103 | Model status, datasets with row counts, agents, Bedrock spend, ship bar | M (row counts are ERP tables) | context.tsx:19 |
| 2 | GET /api/datasets/{ds}/summary | 128 | Dataset summary file (operating rates) | E | Overview, Explorer |
| 3 | GET /api/agents | 166 | Agent configs, stats, autonomy ladder | E | Agents |
| 4 | PUT /api/agents/{agent} | 187 | Update one agent's config | E (keys), G (pattern) | Agents |
| 5 | GET /api/proposals | 219 | List proposals with agent, status, sort, date filters | G | Proposals, ProposalCard |
| 6 | GET /api/proposals/{pid} | 239 | One proposal with evidence | G | ProposalDetail |
| 7 | POST /api/proposals/{pid}/decision | 278 | Approve, adjust or dismiss; reads X-Operator header | G | ProposalCard |
| 8 | POST /api/proposals/{pid}/narrate | 374 | Templated explanation, optional LLM wording checked against evidence | G | ProposalCard |
| 9 | GET /api/lessons | 320 | Learned suppression rules | G | Lessons |
| 10 | POST /api/lessons/{lid}/approve | 329 | Second-person approval for an agent-wide lesson | G | Lessons |
| 11 | POST /api/lessons/{lid}/undo | 357 | Remove a lesson | G | Lessons |
| 12 | GET /api/scorecard | 399 | Latest replay scorecard | C | Overview, Scorecard |
| 13 | GET /api/backtests | 407 | Stored walk-forward backtests | C | Scorecard, Explorer, Developer |
| 14 | POST /api/backtests/run | 422 | Start a backtest job | C | Developer |
| 15 | POST /api/replay/run | 437 | Start a full replay job (not called by console) | C | none |
| 16 | GET /api/jobs | 445 | Background job list | G | Developer |
| 17 | GET /api/jobs/{jid} | 450 | Job progress and result | G | none direct |
| 18 | GET /api/products | 459 | Series picker: name, category, revenue, shelf life | M | Explorer |
| 19 | GET /api/forecast | 471 | History, TimesFM quantiles (q10, q50, q90), baselines, actuals, truth, events, what the model saw | C (core) | Explorer, Developer |
| 20 | GET /api/truth/events | 594 | Labelled twin events (not called by console) | C | none |
| 21 | GET /api/briefing | 603 | Owner briefing for a date | E | Briefing, Overview |
| 22 | GET /api/tenant | 696 | Economics, rollout phase, learning log | E | Rollout |
| 23 | POST /api/tenant | 711 | Change economics, go-live date | E | Rollout |
| 24 | GET /api/policy/frontier | 732 | What each alert budget buys | C | Rollout |
| 25 | GET /api/policy/whatif | 754 | Expected value of alerts at an acceptance rate | M | WhatIf |
| 26 | GET /api/readiness | 777 | Replay one agent on the tenant's recent history | M | Rollout |
| 27 | GET /api/drift | 785 | Forecast calibration by volume class, last eight weeks | C | Rollout |
| 28 | POST /api/tenant/calibrate | 791 | Calibrate from recorded candidates; 404 if none | C | none |
| 29 | POST /api/tenant/learn | 800 | Turn approvals and dismissals into bounded priority weights | E | none (listed in Developer table only) |
| 30 | POST /api/time/step | 809 | Advance the clock, running enabled agents each day | G | TimeMachine |
| 31 | POST /api/time/reset | 863 | Clear replay state and live proposals | G | TimeMachine |
| 32 | GET /api/time/state | 873 | Cursor and test window | G | TimeMachine |
| 33 | GET /api/reveal?codes= | 884 | Map pseudonym codes to real names (local only) | E, privacy-sensitive | context.tsx:35 |

Note on route 33: the handler at `main.py:884-887` has no token parameter. The project CLAUDE.md says `/api/reveal` is token-guarded. The only visible guard is the Host and Origin middleware (`main.py:36-40` comment). Verify before reusing this pattern.

---

## 6. Risks for a rewrite

1. **One global context holds everything** (`context.tsx:5-39`). Dataset, lens, dark mode, reveal and the revealed-name map share one provider. The `value` memo includes `revealed` (37), so every consumer re-renders when a name is revealed. A chat app needs a conversation store. Split it, and do not extend this provider.

2. **No caching and no streaming in the API layer.** `api()` reads the whole body as JSON (`api.ts:34-42`). `useApi` only ignores stale responses (`api.ts:53-58`). Every page refetches on mount. Streaming chat tokens needs `fetch` with a `ReadableStream` or SSE, and the wrapper cannot do that. `main.tsx:9` uses `StrictMode`, which double-runs effects in development. The current code tolerates this; a chat stream would not.

3. **Reveal sends codes in the query string** (`context.tsx:35`: `/api/reveal?codes=...`). These are pseudonyms, not real names, but the project rule is no personal or sensitive data in URLs. Prefer POST with a body. See also section 5, route 33, on the missing token check.

4. **Confirmed date bug (time-zone dependent).** `TimeMachine.tsx:20-22` builds local midnight, adds days, then serialises with `toISOString()`. This was run with Node under `TZ=Australia/Sydney`:
   - Starting from `2026-03-05`, "+1 day" sends `2026-03-05` again. The cursor does not advance.
   - Under `TZ=UTC` the same code sends `2026-03-06`.
   - `BandChart.tsx:77` builds tick labels the same way. In Sydney the first tick for a series starting `2026-03-05` reads `4 Mar`.
   - `Explorer.tsx:35` uses the same parse-then-`toISOString` pattern. Not run, so treat as a likely issue.
   Fix: use one date type end to end (ISO date strings with local-date formatting, or UTC everywhere), and add a test that runs under a positive-offset zone.

5. **Build and serving coupling.** FastAPI mounts `console/dist/assets` only if `dist` exists at import (static block near the end of `main.py`, around lines 889-898). Its SPA fallback answers any unknown GET, including unknown `/api/...` paths, with `index.html`, which hides 404s. The dev proxy covers only `/api` (`vite.config.ts:10`). Streaming endpoints or a websocket need extra proxy config.

6. **Tailwind 4 CSS-first config.** There is no `tailwind.config`. The token file `index.css` is the single source (`:root` at 12-58, `.dark` at 60-85, `@theme inline` at 87-114). Keep that file as the one place for tokens, and keep the `@custom-variant dark` line (6).

7. **Dead code and stale vocabulary.** `Sheet`, `Sparkline`, the four expiry tokens, and `@radix-ui/react-dialog`. Also `Agents.tsx:108` still says "The chance and dollar gates were removed on purpose", while `Agents.tsx:106-115` still shows "Thresholds (advanced)" for the old model. The UI is partly migrated to the DESIGN_REVIEW target (single score and budget, `Rollout.tsx`) and partly not.

8. **Accessibility gaps.** The chart is hover-only (`BandChart.tsx:54-61`) and its SVG has no accessible name (52). The proposal chart has no table fallback (`ProposalCard.tsx:81`). Good parts to keep: skip link (`App.tsx:43`), focus ring (`index.css:122`), `aria-live` readout (`BandChart.tsx:87`), table fallback on Explorer (`Explorer.tsx:105-115`), reduced-motion rule (`index.css:132`).

9. **Quality gates are thin.** Only `tsc --noEmit && vite build` (`package.json:8`). No lint, no test script, no component tests. Add at least tests for chart scale and path builder maths, and for the date helpers (risk 4).

10. **Doc drift to watch.** `docs/DESIGN_REVIEW.md` (section 2) describes a single alert score, tenant-economics dials, a learning loop and a readiness backtest. `Rollout.tsx` implements the economics, frontier, readiness and learning log. `Agents.tsx` still describes the removed gates. Align the rewrite's copy and state with DESIGN_REVIEW section 2 rather than the current console.

11. **Dataset identity is baked in.** `context.tsx:20` defaults to `"twin"`, and every route takes `dataset` (for example `main.py:166`, `219`, `471`). A general Prophet needs a workspace or tenant concept and a series-source abstraction. Renaming in the UI alone will not do it.
