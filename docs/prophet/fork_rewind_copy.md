# Fork, Rewind and Copy at a conversation checkpoint: how Claude does it, and what Prophet takes

Method (2026-11-27): REA (`mcp__rea`, `rea-agents` 6.10.1) on two shipped artifacts, then a static read of the
extracted JavaScript. Nothing was executed.
- **Claude Code 2.1.304**, `/opt/homebrew/Caskroom/claude-code/2.1.304/claude` (Bun standalone, sha256 `b9ac5374…78e6a`):
  `open_binary`, `extract_artifact`, then search and read of the chunks that hold the rewind dialog (`QBe`), the file
  history (`wJt`, `bIt`, `nqo`, `mwe`) and `/branch` (`chunk-7pc42chj`).
- **Claude desktop app** (`/Applications/Claude.app`, `app.asar` sha256 `1aecd637…df863ce`): `extract_artifact` with
  `integrity_policy: record-and-continue` (the declared digest of the unpacked `claude-native-binding.node` no longer matches
  the file, which is what re-signing does; only JavaScript text was read). The main-process chunk `index.chunk-B7Waj9PP.js`
  holds `SessionRewinding` (`canRewindTo`, `planRewind`, `rewindSession`, `undoRewind`, `switchRewindMark`) and
  `forkSession`.

Labels: **observed** is read from code. **Inferred** is a conclusion from the code that no line states. The desktop
renderer (the chat page with the three icons) is not in the asar, so the icons, their order and their tooltips are
**not observed**; they follow the brief. The bridge names behind them are observed.

## 1. What each action does

### Copy (not observed in code; inferred)
Copy on a user message is a clipboard write of that message's text. It touches no session state, so no other design is
needed. In the desktop transcript the sent text can carry attachments; copying the text does not copy them.

### Rewind: return the same conversation to the point before a user message
**Observed, Claude Code.**
- The Rewind dialog lists the session's user messages, the newest as "(current)". Choosing one offers: restore code and
  conversation, restore conversation, restore code, summarize from here, summarize up to here, never mind. When file
  checkpointing is off only "restore conversation" is offered and the dialog says it will "restore and fork".
- Conversation restore (`rewindConversationTo`): find the message in the transcript; `kept = messages[0..i)`,
  `sliced = messages[i..]`. The in-memory transcript is cut and replaced with `kept`. File tracking for the sliced
  messages is evicted, turn state is reset, the pending tool calls outside `kept` are discarded, and the chosen message's
  text (and pasted images) goes back into the prompt box. Telemetry records how many messages were removed.
- Code restore is separate (`wJt`). Every user message gets a file-history snapshot (`mwe` kind `snapshot`): a map from
  each tracked path to a backup file name, or `null` when the file did not exist yet. Restoring walks the tracked paths:
  a backup is copied back, a `null` entry deletes the file the later turns created. Paths that are links or not regular
  files are skipped and counted (`skippedLinks`); a restore where every differing file failed raises "No files were
  restored". A dry run (`bIt`) gives the file list and insertions/deletions for the confirmation screen. Only files
  changed by the editing tools are tracked: the dialog warns that rewinding "does not affect files edited manually or
  via bash". Snapshots are capped in number (oldest trimmed).

**Observed, desktop app.**
- `planRewind` refuses (returns null) when there is no CLI session yet, the session is bound to a history rewrite it
  cannot allow, the target has no assistant fork point before it, the message has no text to put back, or the target is
  on a **dead branch** (a message no longer on the active chain; logged as `rewind_rejected_dead_branch`).
- `rewindSession` waits until the session has been quiet for 300 ms, holds new sends (`rewindUnderWay`), stops the
  session process, and sets `pendingRewindTo = forkPoint`. It records a **pending rewind marker**: forkPoint, the leaf
  before the rewind (`leafBefore`), `prefill`, a copy of the transcript (`copy`), context rows kept, crash records dropped.
  The chain leaf moves to the fork point; the message buffer is cleared. The result returns `{prefill, assistantUuid}`
  and the page puts the prefill in the composer.
- **The cut is applied at the next send** (`transcriptCuts`), and until then it is **undoable**: `undoRewind` restores
  the saved copy (`uncut`), and refuses with "no pending rewind", "a send won the mutex" or an unreadable transcript.
  `switchRewindMark` moves between earlier rewind marks; a mark whose transcript is gone keeps its label only.
- Files go through the SDK call `query.rewindFiles(userMessageId, {dryRun})` and are refused when checkpointing was off
  at start, when an earlier rewind still waits for the next message, when a side chat works in the parent's folder, or
  when the session is mid-turn ("Session is busy"). The presumed-settled timer is 3 minutes. After a code-only restore
  the model is told once that the files went back to an earlier point and that edits it described may no longer be on
  disk, so it should check `git status` and not redo them unless asked.

**Inferred.** The transcript file is append-only with a `parentUuid` chain, so a rewind abandons a branch rather than
deleting it; that is why a "dead branch" exists, why `copy`/`leafBefore` can restore it, and why the CLI reports
"messages removed" as a count and not as a deletion. The dropped messages are therefore recoverable on disk.

### Fork: a new conversation from the checkpoint, original untouched
**Observed, Claude Code `/branch`** ("Create a branch of the current conversation at this point"; `/fork` is a different
command that starts a background agent). `createFork` makes a new session id, reads the source transcript, and keeps
only records whose uuid is in the *current* message list (so dead branches and side-chain records are not carried). Each
kept record is rewritten with the new `sessionId`, a fresh `parentUuid` chain, and `forkedFrom: {sessionId, messageUuid}`.
Content-replacement records (large tool results stored elsewhere), memory-mode records and a relocated working folder are
copied. The file is written private (mode 0600). The title is the first prompt (100 characters) plus " (Branch)" or
" (Branch N)", numbered by what already exists. The message says which session you are in and how to return. Errors:
"No conversation to branch", "No messages to branch". The original is not modified.

**Observed, desktop `forkSession(parent, newId, cutMessage, cwd, options)`.** Refused when the parent has no transcript,
when nothing has been sent, when the new id already exists or is malformed, and for a remote session into a local folder.
The new session gets its own working copy (a new git worktree from the parent's branch) so the fork's file edits do not
collide with the parent's. The transcript is copied whole, or up to the cut message (`copyTranscriptUntilSettled`);
inherited remote-control bridge pointers and worktree bindings are cleared from the copy. When the **cut is the first
turn** the fork is a **fresh start**: no transcript is copied. A "side chat" (`/btw`) reuses `forkSession` and adds
refusals when a copy could re-arm scheduled jobs. Model state is forked with it. The parent is not modified.

**Inferred.** The cut is *before* the chosen message (first turn means empty history), and the chosen text is offered back
in the new composer as a draft, the same as Rewind. Rewind and Fork share one idea: a checkpoint is a message id, and the
state at it is "everything before this message".

## 2. State: what is cut, copied or kept

| State | Rewind | Fork | Copy |
|---|---|---|---|
| Messages after the checkpoint | Dropped from the active chain; kept on disk, undoable until next send | Never copied | n/a |
| Messages before it | Kept in place | Copied to a new session id | n/a |
| Tool results | Go with their messages | Copied with them (large ones through content-replacement records) | n/a |
| Run state (running process, pending calls) | Process stopped; pending calls outside the kept chain discarded | New process; nothing running carried over | n/a |
| Files | Only on request, from per-message snapshots; bash and manual edits not covered | Not copied; the fork gets its own working copy | n/a |
| Composer | Chosen text restored as a draft | Inferred: same | Text to clipboard |
| Original | Same session, history shortened | Untouched | Untouched |

Edge cases seen in code: a checkpoint on a dead branch is refused; a busy or mid-turn session is refused or waits; a
message with no text cannot be a rewind point; a second rewind while one is pending is refused for files; images pasted
into the message are put back with their ids; deletion of the on-disk transcript degrades marks to labels; the cut at the
first message is an empty conversation; fork id collisions are refused.

## 3. What Prophet takes

Prophet's conversation is a **notebook**; each **turn is a mission**; the user message is the mission's `goal`. A quick
forecast is a notebook-less mission, a conversation of one message. The runtime's replay rules (`CLAUDE.md`) decide the
details; Claude's append-only transcript becomes Prophet's flagged rows.

**Checkpoint** = "before turn N" (the message's turn).

**Rewind (notebook).** In one transaction the turns from N on get `dropped_by = <rewind id>`; each gets a
`mission.dropped` event; the table versions those turns saved are flagged the same way; a `rewinds` row records the
cut, the draft text and what was hidden. Nothing is deleted: events, steps, tool calls and questions of the dropped
turns stay and are returned by `GET /api/notebooks/{id}/rewinds`. Every read that builds the conversation (turn list,
next sequence number, planner context, table list and download, data pack) skips flagged rows. Uploaded sources are never
touched. Like Claude, the cut is **undoable until the next question is sent** (`POST /api/rewinds/{id}/undo`); a
sent question makes it final but still recoverable by hand from the ledger. Refused while a turn is active ("stop it
first"), and for a turn already dropped.
Owner decision (2026-11-27): tables saved by dropped turns are hidden, and Undo restores them.

**Rewind (quick forecast).** The forecast is hidden the way Hide works today (reversible), a `rewinds` row records it,
and the message and its files go back in the box; Undo shows it again.

**Fork (notebook).** A new notebook starts with: the instructions and sharing policy; the sources that existed before the
checkpoint; the table versions saved by the kept turns; and a **copy of each kept turn's ledger** (mission, events, steps,
tool calls, questions, notes) under new ids. Tool-call keys include the mission id, so every copied call is re-keyed
and the keys named inside copied events are rewritten to match. A call stays exactly as it was: `done` stays `done`
with its stored result and is never run again; `started` stays `started`, so a non-idempotent call that a crash left
there is still not retried and the turn keeps `outcome_unknown`; model decisions are the stored step outputs and are
replayed, not asked again. A copied turn that was somehow still active is stored as `interrupted` with its open
questions cancelled. The original notebook is not written to. The message goes into the new composer as a draft.
Run folders (charts, outputs) are **shared by reference**, so deleting a notebook or forecast now keeps a run folder that
another turn's ledger still names. The fork is recorded on the notebook (`forked_from`, `forked_at_seq`) and each copied
turn (`forked_from`).

**Fork (quick forecast).** There is no earlier history, so the checkpoint is the empty conversation: a new quick-forecast
box with that message and its files; the original stays as it is.

**Copy.** Clipboard write of the message text in the page; nothing on the server.

Not carried from Claude: summarize-from-here (Prophet has its own context budget), file restore from snapshots (the
sandbox runs write to their own run folders; uploads are never touched), per-message worktrees (Prophet has no working
copy to fork).

## 4. Limits of this note
- The chat page's React code is not in the app bundle, so button look, order and copy text are the brief's, not observed.
- Which messages the desktop shows buttons on (for example whether queued or slash-command messages get them) is not read.
- Whether `/branch` and the desktop fork differ in where the cut falls is inferred from the first-turn "fresh start" rule.
- Source `included`/`removed` flags are copied as they are now; Prophet does not version them.
