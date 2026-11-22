# Slash and at-mention chips: evidence and design

Prophet's `/` and `@` used to paste plain words into the box. The agent saw words, not the thing. This note records how
shipped agent products do it (read from their binaries and sources) and what Prophet does as a result.

## Evidence (reverse-engineered, 2026-11-22)

**Claude Code 2.1.304** (`/opt/homebrew/Caskroom/claude-code/2.1.304/claude`, sha256 `b9ac5374…78e6a`, Bun standalone,
2729 modules; read with REA `open_binary` and `inspect_artifact`, then string and source search of the embedded payload).
- The sent text stays plain. Picks travel as typed *attachments* beside it: `at_mentioned_files`, `at_mention_reference`,
  `mcp_resources`, `agent_mention(s)`, `skill_mention(s)`, `already_read_file`, `dynamic_skill`, `compact_file_reference`.
- Each attachment is turned into a hidden (`isMeta`) message after the user turn. A skill mention becomes: the message
  contains `/<name>`, which is a skill; if the person asked to run it, call the Skill tool with that name and the
  arguments. An agent mention becomes: the person wants to invoke that agent. A file mention becomes a synthetic Read tool
  call and result; a directory becomes a synthetic `ls`. A file too large to read whole is not attached; the model is told
  to read it in portions. A mention that could not be examined is named and left for the model to read itself.
- Skills use progressive disclosure: the names and descriptions are listed, the body loads on invocation. A user-typed
  `/name` is recorded in the transcript as a command tag (`<command-name>/name</command-name>`).

**Cursor** (`workbench.desktop.main.js`): the box is a Tiptap (ProseMirror) editor with the `mention` extension
(`renderText` gives `${char}${label}`), and prompt-input command kinds `skill`, `subagent`, `command`, `mode`, `action`,
`mention`, `heading`, `divider`. Chips are atom nodes, so Backspace removes one whole.

**Public sources** (haiku fan-out, `research_notes/slash_at_chips/public_docs.md`): OpenCode keeps `{start,end,text}` mention
ranges plus typed `files`, `agents`, `skills` arrays and a `contentEditable=false` span; Zed uses crease widgets and a URI
scheme; Cline keeps `@path` as text and appends `<file_content>` blocks; Codex writes a skill as a markdown link
`[$name](path)`. Claude.ai's own composer is closed; the screenshot named in the request did not arrive, so the chip look
follows the Claude Code, Cursor and OpenCode evidence above.

Common rule across all of them: **keep the sent text readable, carry the pick as structure, resolve it on the harness side.**

## Design

- **Wire form.** A pick is a markdown link with a `prophet:` scheme inside the message:
  `[/Choose model by backtest](prophet:skill/choose-model-by-backtest)`, `[/Accuracy check](prophet:tool/backtest_series)`,
  `[@sales.csv](prophet:file/<id>)`. The label carries its sigil, so stripping the link syntax gives the words the person saw.
  One stored string serves the sent bubble, Retry, Edit, history and replay. Codex and Zed use the same link idea.
- **Server resolves and validates** (`src/prophet_agent/mentions.py`, `MissionService.mentions_context`). The server parses
  the links itself and checks each against what exists; a stale or forged id is plain text, never an error.
  - skill: the playbook body (cut at 6000 characters, at most 3) goes into the planner context as "invoked, follow it".
  - tool: named as requested ("use it, or say why the data made it unsuitable"); the evaluator sees it too.
  - file: attached to the mission when it was not already (quick forecast) or resolved to the notebook source, and named as the
    subject of the question, ahead of the usual file listing.
  The planner, titles, notebook history and the evaluator get the words, never the link syntax.
- **API.** `POST /api/missions` and `/api/notebooks/{id}/turns` take the same `goal` string. The 2000-character limit counts the
  words shown, not the link syntax (stored cap 6000). New `GET /api/skills` and `GET /api/skills/{name}` list and read the
  playbooks, replacing the old front-end starter prompts as the source of "skills" (starters stay as "Quick starts").
- **Editor.** Tiptap (ProseMirror) with two mention nodes (`fileMention` for `@`, `commandMention` for `/`), a custom list
  that keeps Prophet's look, Enter to send, Shift+Enter for a line break, Enter during IME composition left alone, Enter on
  touch screens breaks the line, Backspace removes a whole chip, the "+" menu opens the same `/` list narrowed to skills or tools.
- **Sent message.** `MentionChip`: a link-styled button with an icon; clicking opens a card with the playbook summary and first
  lines, the tool description, or the file's role. Escape and outside click close it.

## Verified

`tests/prophet_agent/test_mentions.py` (parse, plain, playbook loaded, tool requested, file attached, unknown picks plain,
limits and titles, notebook sources, skills endpoints). In the browser against an isolated stub API: `/` filter and Enter,
chips in the box, send, chips in the sent bubble, the card, `@` file list and attach, "+" menu list, Backspace on a chip,
phone width, and the planner's own input showing the playbook, the tool request and the file with no link syntax.
