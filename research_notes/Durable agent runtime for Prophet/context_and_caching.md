# Context compaction and prompt caching for long-running agents (Prophet on NVIDIA NIM)

Scope: (a) compaction strategies and their evidence, (b) cache-friendly request layout and what NIM exposes, (c) implications for an append-only agent loop. Current as of 2026-11-20. Tags: [Verified: URL] = read in a fetched source; [Hypothesis] = my inference, not in a source; [Gap] = not found.

Research notes on the sources: the JetBrains blog page returned HTTP 403, so JetBrains figures come from the arXiv v3 paper. The NVIDIA 1.13.0 docs page redirects to archive.docs.nvidia.com; the content below is from the archive copy. A search for SGLang RadixAttention returned an unrelated arXiv paper (2312.17276, PanGu-pi), so SGLang claims are not verified here. Claude Code auto-compact behaviour was not fetched, so it is a gap.

## 1. Compaction and context management

### Takeaway
The best-evidenced policy is cheap, reversible clearing of old tool outputs first, then summarisation only when the trajectory actually needs it. Summarisation is not clearly better than observation masking and can make trajectories longer. For an agent that must keep going, the summary must be a structured handoff, and the durable state should live in files outside the context.

### Cited findings

**Anthropic guidance (context engineering, 29 Sep 2025)**
- Compaction is "the first lever in context engineering to drive better long-term coherence". Tune it by maximising recall first, then precision. "Overly aggressive compaction can result in the loss of subtle but critical context." — [Anthropic](https://anthropic.com/research/effective-context-engineering-for-ai-agents)
- Tool result clearing is "one of the safest lightest touch forms of compaction": once a tool has been called deep in history, the raw result rarely needs re-reading. — [Anthropic](https://anthropic.com/research/effective-context-engineering-for-ai-agents)
- Structured note-taking: the agent writes notes persisted outside the context window, for example a NOTES.md file or to-do lists. — [Anthropic](https://anthropic.com/research/effective-context-engineering-for-ai-agents)
- Sub-agents return a condensed summary, "often 1,000-2,000 tokens". The post gives no quantitative evidence for the gain. — [Anthropic](https://anthropic.com/research/effective-context-engineering-for-ai-agents)
- The post gives no compaction trigger threshold. — [Anthropic](https://anthropic.com/research/effective-context-engineering-for-ai-agents)
- Claude Code's summary keeps architectural decisions, unresolved bugs and implementation details, and the five most recently accessed files. — [Anthropic](https://anthropic.com/research/effective-context-engineering-for-ai-agents)

**Anthropic API compaction (beta, on-demand and threshold modes)**
- Compaction replaces older turns with a server-written summary. Two beta modes: on-demand (the client requests the summary and swaps the returned block in) and at a token threshold (one parameter on ordinary requests; the API drops what came before the block). On-demand uses beta header `compact-2026-09-04`. Options include keeping recent turns verbatim, background compaction, and a custom summarisation prompt for when the default drops something a later turn needs. — [Anthropic docs](https://platform.claude.com/docs/en/build-with-claude/compaction)
- Context editing (clearing old tool results or thinking blocks by rule, without summarising) is a separate feature. — [Anthropic docs](https://platform.claude.com/docs/en/build-with-claude/compaction)
- Threshold compaction is documented as a parameter on ordinary requests. Its trigger value and default are on the threshold page, which I did not fetch. — [Anthropic docs](https://platform.claude.com/docs/en/build-with-claude/compaction)

**JetBrains Research, "The Complexity Trap" (arXiv 2508.21433 v3; NeurIPS 2025 DL4Code workshop)**
- Setup: SWE-bench Verified (500 instances), SWE-agent, turn limit 250. Observation masking window M=10. LLM-Summary N=21, M=10. Models: Qwen3-32B (thinking and non-thinking), Qwen3-Coder-480B-A35B, Gemini 2.5 Flash (thinking and non-thinking). — [arXiv](https://arxiv.org/html/2508.21433v3)
- Qwen3-Coder-480B: raw agent 53.4% solve, $1.29 per instance; observation masking 54.8%, $0.61; LLM-Summary 53.8%, $0.64. — [arXiv](https://arxiv.org/html/2508.21433v3)
- Gemini 2.5 Flash (thinking): raw 40.4%, $0.56; masking 36.4% (-9.9%), $0.24; LLM-Summary 31.4% (-22.3%), $0.25. The only configuration with a significant solve-rate drop for either strategy. — [arXiv](https://arxiv.org/html/2508.21433v3)
- Both strategies cut cost by more than 50% in most configurations. — [arXiv](https://arxiv.org/html/2508.21433v3)
- Trajectory elongation: LLM-Summary trajectories were 15% longer than raw on Qwen3-Coder and 15% longer than masking on Gemini Flash. The authors suggest summaries act as a reinforcing signal that keeps the agent going. — [arXiv](https://arxiv.org/html/2508.21433v3)
- Hybrid (masking plus summarising with unmasked context, N=43): +2.6 points solve over raw on the 50-instance subset, 7% cheaper than masking and 11% cheaper than summary. — [arXiv](https://arxiv.org/html/2508.21433v3)
- Summary API calls were up to 7.2% of instance cost. Once they are removed, the efficiency gap between the two strategies largely disappears. — [arXiv](https://arxiv.org/html/2508.21433v3)
- Limitations the authors state: one domain (SWE-bench, long verbose tool outputs); both triggers are fixed and non-adaptive; the OpenHands result is initial. — [arXiv](https://arxiv.org/html/2508.21433v3)

**Manus (Yichao Ji, 18 Jul 2025)**
- Compression is risky when irreversible. Manus treats the file system as the ultimate context. Web page content can be dropped if its URL stays in context, and a document if its path stays. "Restorable compression". — [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- Do not add or remove tools mid-run. Mask logits instead (constrain choice at decode time). Use consistent tool-name prefixes (browser_, shell_) so groups can be restricted. — [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)

**OpenAI (Responses API compaction)**
- Server-side mode: set `context_management` with a `compact_threshold` on a normal `/responses` call; the server compacts when the rendered token count crosses it. Standalone `POST /responses/compact` returns a compacted window you pass to the next call (stateless, ZDR-friendly). — [OpenAI compaction](https://developers.openai.com/api/docs/guides/compaction) and [context management](https://developers.openai.com/api/docs/guides/context-management) (found via search; not fetched in full)
- A community forum post reports server-side compaction being silently skipped when history contains parallel native shell calls with store=false. Unconfirmed. — [OpenAI community](https://community.openai.com/t/responses-api-compaction-silently-skipped-by-parallel-native-shell-call-history/1389714)
- Summarisation or compaction changes the prefix and resets reuse. — [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)

**Chroma, "Context Rot" (July 2025 technical report)**
- Evaluated 18 LLMs; reliability falls as input length grows even when task complexity is held constant. Degradation accelerates as semantic similarity between target and query falls. — [Chroma](https://research.trychroma.com/context-rot) (secondary summaries only; the primary page was not fetched)

**Not yet verified (fetched or searched without a usable primary source)**
- Letta/MemGPT memory tiers: not searched in this pass. [Gap]
- Claude Code auto-compact thresholds and behaviour: not fetched. [Gap]
- SGLang RadixAttention paper and docs: the search returned an unrelated paper. [Gap]

### Inferences
- Order of operations for Prophet: (1) clear or mask old tool outputs and keep their file paths or handles; (2) write durable state to files; (3) summarise only at a threshold. This follows Anthropic's "lightest touch first", Manus's restorable compression, and JetBrains's finding that masking is at least as good as summarising.
- Summaries should be used sparingly because they can lengthen trajectories. Trigger on token budget, not on turn count.
- Sub-agents are useful for isolating large exploration work (for example, spreadsheet or forecast scans) that returns a 1–2k token digest.

### Gaps
- No primary-source evidence was found for specific compaction trigger percentages in NIM-hosted models. Thresholds should be set from Prophet's own measurements (see section 3).
- The JetBrains results are for SWE-agent on coding tasks. Their transfer to forecasting or spreadsheet missions is untested. [Hypothesis]
- The "what a summary must contain" list below is my design proposal, not a cited standard.

#### Proposed compaction summary schema (design proposal, [Hypothesis])
A summary used to hand off to a continuing agent should be structured, not prose. Suggested sections:
- Goal and success criteria (verbatim from the mission, not paraphrased)
- Hard constraints and user instructions (verbatim, with source turn)
- Decisions made and why (with rejected alternatives)
- Artefacts: file paths, URLs, table names, run IDs, and what each contains
- Tool state: which tools were called, key result values, and where full outputs are stored
- Failed attempts and why they failed (so they are not retried)
- Open questions and next step
- Confidence labels: verified vs assumed vs estimated

Failure modes to guard against: lost constraints (keep the verbatim instructions outside the summary, pinned in the system prompt); summary drift (re-summarising a summary compounds errors, so keep the originals on disk and always summarise from the source transcript); aggressive compaction dropping subtle context (Anthropic); summaries that read as completion signals and extend trajectories (JetBrains).

---

## 2. Prompt caching: mechanics, invalidation and what NIM exposes

### Takeaway
Automatic prefix caching is the same mechanism across vLLM, SGLang and NVIDIA NIM: a shared, byte-identical prefix is reused from KV cache. The layout rule is therefore identical everywhere: stable content first, append-only history, deterministic serialisation, volatile data last. What NIM actually exposes to clients is not confirmed by public NVIDIA documentation. The project's own observation in NIM.md section 7 is the only evidence for usage fields, and it needs a live check before the design depends on it.

### Cited findings

**vLLM automatic prefix caching**
- Each full KV block's hash covers the parent block's hash, the block's tokens, and extra keys (LoRA ID, multimodal hashes, cache salt), so a block is identified by its full prefix. Only full blocks are cached. Eviction is LRU via a free queue. — [vLLM design doc](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- Default hash algorithm is sha256 as of v0.11; `--prefix-caching-hash-algo` offers sha256_cbor, xxhash and xxhash_cbor. — [vLLM design doc](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- The doc does not describe how hits are reported. [Gap] — [vLLM design doc](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- Older vLLM docs describe builtin hashing as default and note collision risk; the current doc differs, so check the version in use. — [vLLM v0.8.x docs](https://docs.vllm.ai/en/v0.8.5/design/v1/prefix_caching.html)

**NVIDIA NIM KV cache reuse (self-hosted NIM container docs, version 1.13.0)**
- Enable with `NIM_ENABLE_KV_CACHE_REUSE=1`. The page says the benefit applies when more than 90% of the initial prompt is identical across requests and only the final tokens differ, with about 2x time-to-first-token speedup from the second request. — [NVIDIA NIM 1.13.0 (archive)](https://archive.docs.nvidia.com/nim/large-language-models/1.13.0/kv-cache-reuse.html)
- Host offloading is optional (`NIM_ENABLE_KV_CACHE_HOST_OFFLOAD`), TensorRT-LLM backend only, default host memory fraction 0.1. — [NVIDIA NIM 1.13.0 (archive)](https://archive.docs.nvidia.com/nim/large-language-models/1.13.0/kv-cache-reuse.html)
- The page does not give block size, eviction policy, TTL, or whether responses report cached tokens. — [NVIDIA NIM 1.13.0 (archive)](https://archive.docs.nvidia.com/nim/large-language-models/1.13.0/kv-cache-reuse.html)
- Search found no NVIDIA NIM documentation of `usage.prompt_tokens_details.cached_tokens`. The field name appears in DeepInfra's docs (a different provider). — [DeepInfra](https://docs.deepinfra.com/chat/prompt-cache-retention.md) (search result only)

**Project observation (NIM.md section 7, [V] by its author, 2026-11-20)**
- `usage.prompt_tokens_details.cached_tokens` was 32 and 48 (of about 50 prompt tokens) on poolside/laguna-xs-2.1, and 0 of 23 on nemotron-3-ultra for a trivial prompt. — /Users/liamle/Downloads/prophet/NIM.md section 7

**Disagreement and what it means**
- NVIDIA's published KV-reuse guidance is for self-hosted NIM containers, controlled by env vars. The hosted integrate.api.nvidia.com endpoint is not covered by these docs, so it is not clear whether the hosted service has the same behaviour. NIM.md's observation shows caching occurs on at least one hosted model, but the 0-of-23 result on Ultra is not explained by any public doc. [Hypothesis] The difference could come from prompt length vs block size (a 23-token prompt may be below any cacheable block), per-model cache configuration, or the hosted gateway's routing. Each is untested.
- NIM.md's "Automatic, model dependent" claim is consistent with the public docs only as far as "reuse happens on shared prefixes". The per-model claim has no public support.

**Anthropic prompt caching (reference)**
- Prefix order is tools, then system, then messages. A `cache_control` breakpoint caches everything up to and including that block. Up to 4 explicit breakpoints per request; each lookup walks back at most 20 blocks. Default TTL 5 minutes, refreshed on each hit; 1-hour TTL via `"ttl": "1h"`. Write costs 1.25x (5 min) or 2x (1 h); reads cost 0.1x on standard models. Minimum cacheable length varies by model (512 to 4,096 tokens in the table). — [Anthropic docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Invalidation: changing tool definitions invalidates all levels; changing the system prompt invalidates system and message levels; changing tool_choice, images, or thinking config invalidates the message level. Don't put breakpoints on content that changes every request. — [Anthropic docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Usage fields: `cache_creation_input_tokens`, `cache_read_input_tokens`, `input_tokens`. — [Anthropic docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

**OpenAI prompt caching (reference)**
- Automatic and on by default for supported models. Cache reads are discounted up to 95%. Earlier models report `cached_tokens` in 128-token steps; GPT-5.6 and later use exact boundaries with a 1,024 visible-token minimum. Retention options are `in_memory` (about 5–10 min idle, up to 1 h) and `24h`; newer models use `prompt_cache_options.ttl` of 30m or 12h. Changes to model, tools, text.format, reasoning.effort, or context_management (compaction) reset reuse. Usage is under `usage.input_tokens_details.cached_tokens`. The doc does not show `prompt_tokens_details`. — [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)

**Manus (KV-cache design)**
- "Even a single-token difference can invalidate the cache from that token onward." A second-precise timestamp near the start of the system prompt is a common cause. Keep context append-only and serialise JSON deterministically. Insert breakpoints at least at the end of the system prompt, allowing for expiry. — [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- Cache hit rate is the key production metric (Manus's framing, as reported). — [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)

### Inferences
- The same rules apply to vLLM, SGLang (not verified here), NIM self-hosted and hosted NIM, Anthropic and OpenAI: reuse depends on an exact byte-identical prefix. Invalidators are anything before the reuse point that changes: system prompt edits, tool schema changes, model switch (NIM fallback to another model loses all cache), reasoning or thinking config changes, and compaction (it rewrites the prefix).
- Model fallback is a cache cost. NIM.md's fallback chain means a failed primary request resets the cache on the next model. Keep the same primary model for a mission where possible.
- Thinking-mode toggles change the template and so change the prefix; keep thinking flags constant per mission.

### Gaps
- Hosted NIM cache semantics are not documented publicly: block size, minimum length, TTL, eviction, and whether usage reports cached tokens. [Gap]
- The 0-of-23 result on Ultra is unexplained. [Gap]
- SGLang RadixAttention specifics (radix tree, LRU, granularity) were not verified. [Gap]

---

## 3. Implications for Prophet (append-only agent loop on NIM)

### Takeaway
Use a fixed prefix (system, role plan, tool schema, mission brief), then an append-only transcript where each tool result is written once and never rewritten, plus a clearing step that replaces old large tool outputs with short stubs (handle, path, row count, key figures) at deterministic points. Summarise only when the token budget is near the limit, and write the summary as a structured handoff. Measure cache hit rate from the live endpoint before tuning thresholds.

### Recommended message layout (per request)
1. System prompt: fixed text, no timestamps, no run IDs, no random values. Tool schemas serialised with sorted keys and fixed order.
2. Mission brief and pinned constraints, verbatim.
3. Durable state pointer: a short file path to NOTES or state, not the content.
4. Transcript: append-only. Each assistant turn and tool result is appended once and never edited.
5. Volatile per-turn data (current time, new user message) at the very end.

### Compaction policy (proposed, [Hypothesis] on thresholds)
- Stage 1, continuous: replace tool outputs older than N turns with stubs that keep the file path or handle. Because this edits old content, it breaks the cache from that point. Do it in batches at a fixed threshold, not every turn, so the cache survives between batches. (The batching is a design inference from Manus's cache advice.)
- Stage 2, threshold: when the transcript passes roughly 60–70% of the input budget, write a structured summary (schema in section 1) and restart from prefix + summary + recent turns verbatim. Keep the last few turns intact (Anthropic's "keep recent turns" option).
- Stage 3: sub-agents for bounded exploration returning a 1–2k token digest.
- Trigger on tokens, not turns. Compaction resets the cache, so budget one cold request per compaction.
- Keep full transcripts and tool outputs on disk so summaries can be regenerated from source (avoids drift).
- Prophet's input budget per NIM.md is 200,000 tokens per role plan. Use this as the basis for thresholds, [Hypothesis].

### How to measure cache hit rate
- For each request, log `prompt_tokens`, `cached_tokens` (if the field is present), and the model ID. Hit rate = sum(cached) / sum(prompt).
- First confirm the field exists. Send two identical 3k+ token requests back to back to the hosted endpoint on each model, then a third with a 1-token change at the start, and compare. This separates "field missing" from "no cache".
- Log the prefix hash (SHA-256 of the system prompt plus tool schema) per request so misses can be correlated with prefix changes.
- Record cost and latency per turn to see whether cache hits reduce time-to-first-token.
- Expect cache effects only above the model-specific minimum length; below it, zero cached tokens is correct.

### Cost and latency expectations
- Caching discounts vary by provider and are not documented for NIM hosted endpoints. [Gap] For Anthropic the cache read is 0.1x of base input (reference only). For OpenAI, up to 95% off input reads (reference only).
- Latency: NVIDIA's own guidance claims roughly 2x time-to-first-token improvement when the prefix is over 90% identical (self-hosted docs, [NVIDIA NIM 1.13.0 (archive)](https://archive.docs.nvidia.com/nim/large-language-models/1.13.0/kv-cache-reuse.html)). Treat as an upper bound from a different deployment, not a Prophet estimate.
- Prophet's own numbers from NIM.md (about 1–5 s per call with thinking off) suggest prefill is not the dominant cost for short prompts; measure it on long transcripts before assuming a large gain. [Hypothesis]

### Open questions to resolve with live tests
1. Does the hosted NIM endpoint return `usage.prompt_tokens_details.cached_tokens` for every model, or only some? (Test with the protocol above.)
2. What is the minimum cacheable prefix on NIM hosted models, and is the 0-of-23 result a length effect?
3. Does a model fallback lose cache, and how long does a prefix stay warm between calls?
4. Does the hosted gateway route by key or by model, and does rotating keys (NIM.md section 4) split the cache across replicas?

### Gaps
- No public documentation for hosted NIM cache TTL, granularity, or eviction. [Gap]
- No published benchmark of summary-based compaction on forecasting or spreadsheet-style agent work. [Gap]

---

## Source list
- Anthropic, Effective context engineering for AI agents: https://anthropic.com/research/effective-context-engineering-for-ai-agents
- Anthropic docs, Compaction: https://platform.claude.com/docs/en/build-with-claude/compaction
- Anthropic docs, Prompt caching: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- JetBrains Research blog (403 on fetch): https://blog.jetbrains.com/research/2025/12/efficient-context-management/
- arXiv 2508.21433 v3, The Complexity Trap: https://arxiv.org/html/2508.21433v3
- Manus, Context engineering for AI agents: https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus
- OpenAI, Compaction: https://developers.openai.com/api/docs/guides/compaction
- OpenAI, Context management: https://developers.openai.com/api/docs/guides/context-management
- OpenAI, Prompt caching: https://developers.openai.com/api/docs/guides/prompt-caching
- OpenAI community, compaction skipped bug report: https://community.openai.com/t/responses-api-compaction-silently-skipped-by-parallel-native-shell-call-history/1389714
- Chroma, Context rot: https://research.trychroma.com/context-rot
- vLLM, Automatic prefix caching: https://docs.vllm.ai/en/latest/design/prefix_caching/
- NVIDIA NIM 1.13.0, KV cache reuse (archive): https://archive.docs.nvidia.com/nim/large-language-models/1.13.0/kv-cache-reuse.html
- Project file: /Users/liamle/Downloads/prophet/NIM.md (section 7 and section 4)
