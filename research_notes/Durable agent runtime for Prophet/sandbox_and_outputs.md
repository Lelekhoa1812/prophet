# Sandbox and outputs for agent-written Python on a local macOS FastAPI app (as of 2026-11-20)

Scope: execution options for model-written Python, the library stack for outputs, and discovery/result patterns for Prophet v1 (Apple silicon, Python 3.12, FastAPI). Tags: [Verified: URL] = read from the cited source in this session; [Hypothesis] = reasoning or unverified claim to test. Searches were partly rate-limited, so several areas (Jupyter kernels, Docker/Podman, e2b/Modal, Pyodide, gVisor/Firecracker, rlimit behaviour on macOS) were NOT researched here and are listed under Gaps.

## Execution options and isolation trade-offs

### Takeaway
No option available on a Mac gives a real security boundary for untrusted model code without a VM. For v1, a fresh per-run subprocess with resource and time limits is an honest guardrail, not isolation. The real isolation options are a VM (Apple `container`, Docker/Podman on a Linux VM) or a hosted sandbox, and those are the path for a server deployment.

### Cited Findings
- Anthropic's code execution tool runs Python and Bash in a containerised environment with no internet access, full isolation from host and other containers, and file access limited to the workspace directory. Its stated limits are 5 GiB RAM, 5 GiB workspace disk, 1 CPU, Python 3.11, and a container that expires 30 days after creation [Verified: https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool]. The docs list a cell-level 90-second wall-clock limit only for programmatic tool calling; the general limit is "maximum execution time" without a stated number [Verified: same URL].
- Anthropic containers are fresh per request unless the container ID is passed back. After about 5 minutes of inactivity a container is checkpointed and can be restored within the 30-day window. Files are returned by writing them to a per-call `$OUTPUT_DIR`, and only top-level files there are captured [Verified: same URL].
- Anthropic bills code execution by container time: a 5-minute minimum, 1,550 free hours per organisation per month, then $0.05 per hour per container [Verified: same URL].
- OpenAI's code interpreter runs in "a fully sandboxed virtual machine". Memory tiers are 1g (default), 4g, 16g and 64g. Containers expire after 20 minutes of inactivity and cannot be reactivated, and Python in-memory state is lost on expiry. The page does not describe the isolation mechanism or network policy [Verified: https://developers.openai.com/api/docs/guides/tools-code-interpreter, via summary fetch].
- Apple's `container` CLI runs Linux containers as lightweight VMs on Apple silicon. The README says it is "designed to run on macOS 26 or later" and "You need a Mac with Apple silicon". It does not state a current release, networking defaults, CPU/memory flags, or a security section [Verified: https://github.com/apple/container]. One third-party guide gives conflicting macOS and version information [Verified: https://www.omgubuntu.co.uk/apple-linux-container-tool-mac-developers, https://learn.arm.com/install-guides/container/ ; conflicts noted].
- `sandbox-exec` (Seatbelt) is marked deprecated in its man page, with Apple's guidance pointing to the App Sandbox. It still works on current macOS releases, and a March 2026 test on Sequoia reported a basic invocation working [Verified: https://gastown.dev/docs/research/macos-sandbox-exec]. One project warns it could be removed in a future release [Verified: https://openparallax.pages.dev/sandbox/sandbox-exec]. The App Sandbox is designed for app bundles, not for a single script [Verified: https://korben.info/en/sandbox-exec-hidden-sandboxing-tool-macos.html].
- RestrictedPython: the documentation says it "is not a sandbox system or a secured environment" [Verified: https://restrictedpython.readthedocs.io/en/latest/]. Its current release is 9.0, which requires Python >=3.11,<3.16 [Verified: https://pypi.org/pypi/RestrictedPython/json].
- Hosted sandbox option: Vercel Sandbox is described in the session's skill catalogue as ephemeral Firecracker microVMs [Hypothesis: skill description only; not fetched].

### Inferences
- Model-written code runs as the user on their Mac, with access to their files, keys and network unless the runner restricts them. The minimum v1 requirement is therefore: a separate working directory, a scrubbed environment (no API keys, no `~`), no network, and hard time and output caps.
- Language-level filtering (AST checks, RestrictedPython) should not be the security control. It is useful only as a lint or to catch accidents. This follows from the RestrictedPython disclaimer above.
- `sandbox-exec` is the only built-in macOS tool that can deny network and file writes for a child process. Because it is deprecated and undocumented for this use, it should be a defence-in-depth layer that can be dropped without breaking the design, not the sole boundary [Hypothesis].
- A persistent Jupyter/IPython kernel gives ChatGPT-style state between calls but adds a long-lived process that is harder to kill cleanly and to isolate. It is a usability feature, not a safety one [Hypothesis]. Anthropic also ties REPL state persistence to a specific tool version [Verified: https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool].
- Apple `container` is the most credible local-VM upgrade for v2 on the Mac, because each container gets a VM boundary, but its resource-limit flags and networking defaults still need to be confirmed from its command reference [Hypothesis].

### Gaps
- Not researched in this session: Jupyter kernel-level isolation, Docker Desktop/Podman on macOS (resource-limit and network flags), e2b, Modal, Pyodide/WASM (pandas and matplotlib support in Pyodide), gVisor and Firecracker on Linux servers.
- Not verified: whether macOS enforces `RLIMIT_AS` (address-space memory limit). Common understanding is that it is not enforced on macOS, which would mean memory caps need an RSS watchdog, but this is unconfirmed here [Hypothesis; test before relying on it].
- Not verified: OpenAI's network policy and exact isolation mechanism; neither is stated on the page fetched.

## Recommended default for Prophet v1 and path to stronger isolation

### Takeaway
Run each agent script in a fresh subprocess inside a per-run directory, using the app's Python 3.12 interpreter with `-I`, a scrubbed environment, a process-group timeout, POSIX rlimits for CPU/file size/open files, an RSS watchdog for memory, capped stdout/stderr, and (where available) a `sandbox-exec` profile that denies network and writes outside the run directory. Label this "guardrail, not isolation" in the UI and docs. Move to a per-run VM (Apple `container` or a hosted sandbox) before any server or multi-user use.

### Cited Findings
- Anthropic's own design separates the sandbox from the application: inputs are uploaded per container, the container has no network, and outputs are captured only from a designated output directory. This is a useful pattern to copy even on a local runner [Verified: https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool].
- OpenAI's code interpreter also captures generated files as citations and downloads them through a separate endpoint, and expires idle containers after 20 minutes [Verified: https://developers.openai.com/api/docs/guides/tools-code-interpreter].
- The Anthropic container pre-installs a fixed library set (pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, seaborn, openpyxl, xlsxwriter, pillow, python-docx, reportlab, pypdf, pdfplumber, sympy, and others) and has no way to install more at runtime. It does not list plotly, altair, kaleido or statsforecast [Verified: same Anthropic URL]. This is a good baseline for choosing Prophet's pre-installed set.

### Inferences
- Recommended v1 runner (local, Mac):
  1. Workspace: `runs/<run_id>/in/` for input copies (read-only where possible), `runs/<run_id>/out/` as the only write location that is captured. Pass the path in via an environment variable, not a hard-coded home path.
  2. Process: `subprocess` with `start_new_session=True` so the timeout kills the whole process group; `-I` flag; a minimal environment (no keys, `HOME` set to the run directory, `TMPDIR` inside the run directory).
  3. Limits: `RLIMIT_CPU`, `RLIMIT_FSIZE` (caps any single written file), `RLIMIT_NOFILE`; wall-clock timeout (for example 60 s default, configurable); RSS watchdog (`psutil`) for memory; capped stdout/stderr (for example 1 MB each, tail kept); cap on total files and total bytes in `out/`.
  4. Network: `sandbox-exec` profile with `(deny network*)` and `(deny file-write*)` except the run directory, if the profile works on the target macOS version. If it fails, run with the network disabled by environment only and say so plainly.
  5. Stream stdout/stderr to the UI over SSE or a websocket, tagged by run ID.
- Path to stronger isolation:
  - v2 local: Apple `container` (VM per container; macOS 26+, Apple silicon), with the input directory mounted read-only and `out/` mounted read-write.
  - Server: gVisor or Firecracker-based runners, or a hosted sandbox (e2b, Modal, Vercel Sandbox), with the same run-directory contract so the application code does not change.
  - Keep the contract the same across all runners (inputs in, a manifest of outputs out, the same limits) so the backend can be swapped.
- Do not offer persistent kernels in v1. Each call is stateless, and the agent re-loads its input from the run directory. A persistent kernel can come later behind a flag.
- Pin the sandbox Python to 3.12 for parity with the host. Anthropic's runtime is Python 3.11, so code that works there may not work unchanged in Prophet [Hypothesis based on the version difference].

### Gaps
- No measured test of the recommended runner on this Mac (rlimit behaviour, `sandbox-exec` profile compatibility, process-group kill). These should be tested before the design is accepted.
- No verified figures for CPU/memory limits in OpenAI's sandbox beyond the memory tiers.

## Library stack for outputs

### Takeaway
Use pandas 3.1 with numpy 2.5 and openpyxl/xlsxwriter for spreadsheets; matplotlib for static PNG/SVG; plotly (with kaleido for static export) or Altair/Vega-Lite JSON for interactive charts; python-docx for Word; reportlab for PDF, with WeasyPrint as an alternative. Every library in the shortlist publishes Python 3.12-compatible releases as of 2026-11-20. Note that the project's `pyproject.toml` currently leaves numpy and pandas unpinned, which conflicts with the brief's pins.

### Cited Findings
- PyPI latest releases checked on 2026-11-20 (requires_python from the PyPI JSON API) [Verified: https://pypi.org/pypi/<name>/json]:
  - numpy 2.5.5 (uploaded 2026-11-15), requires >=3.12
  - pandas 3.1.1 (2026-11-04), requires >=3.11
  - scipy 1.18.1 (2026-08-21), requires >=3.12
  - scikit-learn 1.9.1 (2026-09-10), requires >=3.11
  - statsmodels 0.15.0 (2026-08-27), requires >=3.10
  - statsforecast 2.1.1 (2026-07-16), requires >=3.10
  - matplotlib 3.11.2 (2026-09-11), requires >=3.11
  - plotly 7.1.0 (2026-09-15), requires >=3.8
  - kaleido 1.5.2 (2026-11-18), requires >=3.9
  - altair 6.3.0 (2026-09-15), requires >=3.11
  - bokeh 3.10.2 (2026-11-05), requires >=3.12
  - openpyxl 3.1.5 (2024-06-28), requires >=3.8. The latest PyPI release is from 2024, so check for maintenance status before depending on it for new features.
  - xlsxwriter 3.3.2 (2026-11-19), requires >=3.8
  - python-docx 1.2.0 (2025-06-16), requires >=3.9
  - reportlab 5.1.0 (2026-10-22), requires >=3.9
  - weasyprint 70.0 (2026-09-08), requires >=3.10
  - pypdf 7.1.1 (2026-11-18), requires >=3.10
  - pdfplumber 0.11.10 (2026-06-15), requires >=3.8
  - RestrictedPython 9.0 (2026-11-04), requires >=3.11,<3.16
  - jupyter-client 8.10.0 (2026-08-28), requires >=3.10
- The project's `pyproject.toml` declares `requires-python = ">=3.12"` and lists `numpy` and `pandas` without version pins; the brief's statement that they are pinned at numpy 2.5 and pandas 3.1 is therefore not reflected in this file [Verified: /Users/liamle/Downloads/prophet/pyproject.toml].
- Anthropic's pre-installed set (see above) includes openpyxl, xlsxwriter, python-docx, reportlab and matplotlib, but not plotly, altair or kaleido [Verified: https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool].

### Inferences
- Spreadsheet IO: pandas reads messy sheets with `header=`, `skiprows=`, `usecols=` and `dtype=` settings; merged headers and multi-row headers need a custom normalisation step (this is common practice but not verified here). openpyxl reads and writes formulas as text and does not calculate them; a recalculation engine such as LibreOffice would be needed to get computed values [Hypothesis; formula-recalc claim not verified].
- Charts: matplotlib gives the most predictable static output (PNG/SVG) with no extra runtime. Plotly gives interactive HTML or JSON; its static image export depends on kaleido, so the kaleido version and its browser requirements should be checked before use [Hypothesis; kaleido's runtime requirements not verified]. Altair/Vega-Lite produces declarative JSON that a chat UI can render without Python on the client.
- Documents: python-docx for .docx; reportlab for PDF generation with precise layout; WeasyPrint for HTML-to-PDF if the team prefers HTML/CSS, but it needs system libraries (Pango) that should be checked on the target Mac [Hypothesis].
- For v1, keep the pre-installed set close to what Anthropic ships (pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, openpyxl, xlsxwriter, python-docx, reportlab, pypdf, pdfplumber) so that agent-written code transfers between the local runner and a future hosted sandbox.

### Gaps
- Not verified: kaleido runtime requirements (Chrome or other browser), WeasyPrint's macOS system dependencies, LibreOffice formula recalculation, and merged-header handling details.
- Not verified: Python 3.12 wheel availability on Apple silicon for every package above. The PyPI metadata confirms requires_python, but not wheel coverage; check with `uv pip install` in a clean venv.

## Discovery and returning rich results to the chat UI

### Takeaway
Expose the helpers as a small documented Python module (for example `prophet_tools`) with a `list_helpers()` function that returns name, signature and docstring from `inspect`, plus a generated markdown index that is injected into the agent's system prompt. Return every output as a manifest entry (id, kind, mime, path, bytes, sha256, preview, warnings) so the UI can render previews and download links without re-reading files.

### Cited Findings
- Anthropic and OpenAI both return generated files as separate references (file IDs or citations) and do not inline them into the text reply [Verified: both URLs above].
- Anthropic's docs state that only top-level files in the output directory are captured and that the output listing is what confirms capture [Verified: https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool].

### Inferences
- A helper module is discoverable if its functions have typed signatures and docstrings. `inspect.signature` and `inspect.getdoc` can produce the catalogue at runtime, so the catalogue cannot drift from the code [Hypothesis; standard Python behaviour, not cited].
- Suggested manifest entry: `{id, run_id, kind (table|chart|document|data|log), mime, path (relative to out/), bytes, sha256, preview (first N rows, thumbnail path, or text excerpt), created_at, warnings[]}`. Previews should be generated by the runner, not by the UI, so the UI never opens untrusted files.
- The UI should serve files only from the run's `out/` directory, with a path allow-list, and should set `Content-Disposition` and a safe `Content-Type` for downloads, because the files are model-generated.

### Gaps
- No public example of a specific artefact manifest schema was reviewed. The schema above is a design proposal, not a sourced standard.
- No check of how the existing Prophet API handles file serving; the code was not inspected in this session.

## Summary of open risks for v1
- The local runner is a guardrail. Any design statement must say so.
- `sandbox-exec` is deprecated and its profile behaviour on the target macOS version is untested.
- The memory cap design depends on an unverified macOS rlimit assumption.
- Version pins in `pyproject.toml` do not match the brief.
