# Design: Prophet console

Mode: Operate. Precision and restraint carry the identity; the starfield is the only expressive surface.

## Colour (tokens on :root, all dark-first)

| Token | Value | Use |
|---|---|---|
| --night-0 | #0b0907 | page base |
| --night-1 | #15110d | panels |
| --night-2 | #201a14 | raised rows, inputs |
| --sand-0 | #f1e4c8 | primary text (contrast ≥ 14:1 on night-0) |
| --sand-1 | #b9a682 | secondary text (contrast ≥ 7:1) |
| --sand-2 | #7d6d55 | rules and quiet borders only, never text |
| --ember | #ef7a3a | primary action, focus ring, the agent's live state |
| --ember-soft | #3a2116 | ember-tinted backgrounds for question cards |
| --ok | #8fbf9a | completed |
| --bad | #e0675c | failed, outcome unknown |

## Type

- Display: Cormorant Garamond (self-hosted via @fontsource). Wordmark and page headings only.
- Interface and body: IBM Plex Sans (self-hosted). Measure 65–75ch for prose.
- Data: IBM Plex Mono (self-hosted). Hashes, ids, JSON, byte counts.
- Scale: 0.8125 / 0.9375 / 1.0 / 1.25 / 1.625 / 2.25 rem. Headings balanced.

## Layout

- Desktop (≥ 1024 px): three columns. Missions (260 px), transcript (fluid, max 760 px), artefacts (340 px).
- Tablet and phone (< 1024 px): one column. The missions list collapses behind a disclosure at the top; the artefact
  panel becomes a section below the transcript. Gutter 16 px. No horizontal scroll at 375 px.

## Components

- Buttons: shadcn-style primitives (variant and size via class-variance-authority), 40 px minimum target on touch.
- Question card: a bordered card on ember-soft, one question per block, options as native buttons in a group
  (arrow keys move focus, Enter or Space selects). Multi-select uses checkboxes.
- Transcript rows: step kind as a small label (Plan, Act, Evaluate, Ask), the planner's rationale as the text, and tool
  results in mono. No avatars, no bubbles.
- Status: a dot plus a word. Never colour alone.

## Motion

One authored moment: the starfield drifts slowly and parallaxes with the pointer on desktop; it stops under
prefers-reduced-motion. Nothing else animates except a short fade on newly added transcript rows.

## Background

A canvas starfield (drawn, not an image). Under 120 stars on small screens. A React Bits-style effect implemented in
the project, not copied from the library.
