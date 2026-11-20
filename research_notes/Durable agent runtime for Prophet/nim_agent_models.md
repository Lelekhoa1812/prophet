# NVIDIA NIM hosted models as the brain of a tool-using agent (as at 2026-11-20)

Scope: public documentation and third-party mirrors for NVIDIA NIM hosted chat models (integrate.api.nvidia.com, OpenAI-compatible) for five models listed in `/Users/liamle/Downloads/prophet/NIM.md`. Tags: [Verified: URL] = read on a fetched page; [Search snippet: URL] = seen only in search results, page not fetched; [Third-party: URL] = aggregator or vendor mirror, not primary; [Hypothesis] = my inference, needs a live test.

Research limits: about 17 tool calls. Several primary pages returned empty or 401/404 (listed under Gaps). No API keys were read, printed or stored.

## Tool calling on NIM

### Takeaway
NIM documents OpenAI-style `tools` and `tool_choice` on chat completions, but the per-model support and parser behaviour is only partly documented. The official docs page could not be fetched in full, so model-by-model tool support for the five models is unconfirmed. Live tests are required before the agent loop relies on it.

### Cited Findings
- NIM function calling uses the `tools` and `tool_choice` request fields in the OpenAI-compatible chat completions API. `tool_choice` accepts "none", "auto" or a named tool, and requires `tools` to be set. [Search snippet: https://docs.nvidia.com/nim/large-language-models/latest/function-calling.html]
- The LLM docs say tool calling is automatically enabled for Llama 3.1, 3.2 and 3.3 and Mistral models. The five models in NIM.md are not named in the snippet. [Search snippet: https://docs.nvidia.com/nim/large-language-models/latest/function-calling.html]
- The docs' troubleshooting note says that if the tool-call field is empty but the call appears in message content, the output was not parsed. The suggested fixes are a chat-template change or a different tool-call parser. [Search snippet: https://docs.nvidia.com/nim/large-language-models/latest/function-calling.html]
- Older NIM doc versions (1.2.0, 1.3.0) describe a `parallel_tool_calls` parameter. Newer snippets did not mention it, so its status is unconfirmed. [Search snippet: https://docs.nvidia.com/nim/large-language-models/1.2.0/function-calling.html]
- Vision-language NIM docs say function calling is supported only on specific models, so the model card must be checked for each model. [Search snippet: https://docs.nvidia.com/nim/vision-language-models/2.0.0/function-calling.html]
- Poolside (Laguna XS 2.1 FP8): tool calling is supported. Current vLLM needs PR vllm-project/vllm#47311 and parser `poolside_v1`; older builds use `--tool-call-parser glm47`. This is the self-hosted vLLM path, not NIM. [Verified: https://huggingface.co/poolside/Laguna-XS-2.1-FP8]
- Laguna has native interleaved thinking between tool calls, with preserved thinking. [Verified: https://huggingface.co/poolside/Laguna-XS-2.1-FP8]
- Z.ai GLM-5.3-Flash: function calling is supported. For streaming tool calls the page recommends `stream: true` and `tool_stream: true`. The request and response format is not given on that page and it points to a separate function-calling guide. [Verified: https://docs.z.ai/guides/llm/glm-5.3-flash]
- Third-party hosts (OpenRouter via MCP Playground, Vercel AI Gateway, Cloudflare) list tool and tool_choice for GLM-5.3-Flash, and MCP Playground says it accepts parallel tool calls. [Third-party: https://mcpplaygroundonline.com/models/glm-5-3-flash; https://vercel.com/ai-gateway/models/glm-5.3-flash]
- Nemotron 3 Ultra: third-party mirrors describe the model as built for agentic workflows and tool use. No tool-calling format or benchmark was verified from a primary source. [Third-party: https://haimaker.ai/models/nvidia/nemotron-3-ultra-550b-a55b; https://tech-insider.org/nvidia-nemotron-3-ultra-open-weight-2026/]

### Inferences
- The `tools` request on NIM is probably the same OpenAI shape as in NIM.md section 3, so the NIM.md wire body plus `tools` and `tool_choice` is a reasonable starting point. [Hypothesis]
- Whether tool calls come back in `tool_calls` or as text in `content` depends on the server-side parser for each model. The empty-field pitfall in the NVIDIA docs makes a parser check mandatory. [Hypothesis]
- NIM.md says `response_format` is omitted when `tools` are present. This is a design choice in NIM.md, not a documented NIM limitation. Whether `tools` and `json_schema` can be combined on the same request is untested. [Hypothesis]

### Gaps
- Could not fetch the full NIM function-calling page, so the per-model support list and parser names for the five models are unknown.
- `parallel_tool_calls` support on current NIM versions is unconfirmed.
- Streaming tool-call delta format on NIM (delta.tool_calls with index and id) is not documented in anything I retrieved. NIM.md section 5 says to normalise deltas by stable `id` and `index`, which is the OpenAI convention, but this is not verified against NIM.
- Interaction between reasoning mode and tool calls on NIM is not documented.

## Reasoning and thinking controls

### Takeaway
`chat_template_kwargs.enable_thinking` is confirmed as the switch for Nano Omni and Laguna in third-party and vendor docs. Z.ai's first-party docs say GLM-5.3-Flash thinking cannot be disabled, which conflicts with the NIM.md plan to turn GLM thinking off. This must be tested on NIM.

### Cited Findings
- Nemotron 3 Nano Omni reasoning: `enable_thinking` controls reasoning. Serving guides pass it via `chat_template_kwargs`, e.g. `{"enable_thinking": True}`. [Third-party: https://recipes.vllm.ai/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16; https://lmsysorg.mintlify.app/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni]
- An SGLang cookbook says Nano Omni has two modes, reasoning ON (default) and OFF. The NVIDIA build example sets `enable_thinking` True alongside a `reasoning_budget` of 16384. [Third-party: https://lmsysorg.mintlify.app/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni; Search snippet: https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning?nim=hosted]
- Fireworks says reasoning is supported for text and image inputs only, and that `enable_thinking: false` should be set for video and audio. [Third-party: https://fireworks.ai/models/fireworks/nvidia-nemotron-3-nano-omni-30b-a3b]
- Nano Omni parameter counts conflict across sources (31B total / 3B active; 33.0B; 30B total / 3.5B active). [Third-party: https://dell.huggingface.co/models/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8; https://www.jetson-ai-lab.com/models/nemotron-3-nano-omni]
- Laguna XS 2.1 FP8: thinking is supported and can be enabled or disabled per request with `enable_thinking`. [Verified: https://huggingface.co/poolside/Laguna-XS-2.1-FP8]
- Z.ai GLM-5.3-Flash: `thinking.type` accepts only `enabled`. The page says "thinking cannot be disabled". It recommends `thinking.clear_thinking: false` and `reasoning_effort: max`. No default is stated. [Verified: https://docs.z.ai/guides/llm/glm-5.3-flash]
- Z.ai recommended sampling for GLM-5.3-Flash: `temperature: 1`, `top_p: 0.95`. [Verified: https://docs.z.ai/guides/llm/glm-5.3-flash]
- Third-party reviews conflict on whether GLM-5.3-Flash thinking is always on or can be configured through `reasoning_effort`. [Third-party: https://codepick.dev/en/guides/glm-5-3-flash-guide/; https://www.itechguides.com/how-ai-teams-should-evaluate-glm-5-3-flash/]
- Nemotron 3 Ultra thinking controls: not found in any fetched page. NIM.md states `enable_thinking=false` cuts latency from 25 to 60 s down to 2 to 6 s, which is the team's own measurement. [Unverified against a primary card]

### Inferences
- NIM.md's GLM plan (sending `chat_template_kwargs.enable_thinking=false` plus `reasoning_effort:"low"`) conflicts with Z.ai's first-party statement that thinking cannot be disabled. NIM may expose a different wrapper, so the NIM behaviour must be tested live. The NIM.md note that GLM needs `reasoning_effort:"low"` conflicts with Z.ai's recommended `max`. [Hypothesis]
- NIM.md lists Laguna as having "no thinking mode" (n/a for thinking). The HF card says thinking can be toggled with `enable_thinking`. Either the NIM-hosted build differs from the HF model, or NIM.md's note is wrong. Test before relying on either. [Hypothesis]
- Where a model emits thinking as `<think>` in `content`, NIM.md's stripping logic in section 3.2 is required. Whether NIM returns a separate `reasoning_content` field is not verified.

### Gaps
- No primary Nemotron 3 Ultra or Nano Omni card text was retrieved. Thinking controls and defaults for those models are unconfirmed on NVIDIA's own pages.
- Whether NIM returns `reasoning_content` as a separate field is unknown.
- GLM `thinking.type`/`reasoning_effort` behaviour through NIM is unknown.

## Structured output

### Takeaway
NIM.md section 3.1 is the only source-backed statement on NIM strict JSON-schema behaviour. The official structured generation page returned no text, so the supported option set and keyword list could not be confirmed.

### Cited Findings
- NIM.md section 3.1 records that NIM rejects `uniqueItems`, `minItems` and other keywords, names the offenders in a 400 body (`Unimplemented keys`), and that `json_object` can emit `null` for enums on Nano. This is team observation, not documentation. [NIM.md section 3.1, local file]
- The NVIDIA structured generation page at https://docs.nvidia.com/nim/large-language-models/latest/structured-generation.html returned empty content on fetch. [Fetch returned no text]

### Inferences
- Combining tools with a schema-constrained final answer is best done as two passes (tool loop, then a schema call), as NIM.md already does. A single request combining `tools` and `response_format` is untested. [Hypothesis]

### Gaps
- Official list of supported response_format and guided_json (nvext) options is unconfirmed.
- Official JSON-schema keyword support list is unconfirmed; NIM.md's list is observational.

## Context, output limits, latency, rate limits, and terms

### Takeaway
Model-level context and output figures are mostly third-party. NVIDIA does not publish free-tier rate limits per model, and the Trial Terms as quoted by secondary sources restrict use to testing and evaluation. Multi-key rotation to multiply quota is not shown to be permitted and should be treated as not permitted until someone reads the primary terms.

### Cited Findings
Context and output limits:
- Nemotron 3 Ultra: 550B total MoE, about 55B active, 1M context. Some providers list up to 66K output tokens. [Third-party: https://haimaker.ai/models/nvidia/nemotron-3-ultra-550b-a55b; https://tech-insider.org/nvidia-nemotron-3-ultra-open-weight-2026/] NIM.md uses 1,048,576 context and 65,536 max output (team-observed). The official card was not retrieved.
- Nemotron 3 Ultra release date: several sources say June 4, 2026. [Third-party: https://haimaker.ai/models/nvidia/nemotron-3-ultra-550b-a55b]
- Laguna XS 2.1: 33B total, 3B active; 262,144 context. Max output not stated on the HF card. NIM.md uses 32,768 (team-observed). [Verified: https://huggingface.co/poolside/Laguna-XS-2.1-FP8]
- GLM-5.3-Flash: 1M context, 128K max output. NIM.md uses 32,768 for NIM, which may be a NIM-side cap. [Verified: https://docs.z.ai/guides/llm/glm-5.3-flash]. One mirror gives 1.31M context and 131K output. [Third-party: https://mcpplaygroundonline.com/models/glm-5-3-flash]
- Nano Omni: 256K context per NIM.md. Not checked against a primary source.

Rate limits and quota:
- NVIDIA staff said in April 2025 that NVIDIA does not publish limits for each model, and that the limits "only apply to the APIs which are for trial experiences". Staff pointed to paid or hosted options for higher use, including NVIDIA AI Enterprise and hosted NIM providers. [Verified: https://forums.developer.nvidia.com/t/model-limits/331075]
- The 40 RPM figure is user-reported on NVIDIA forums (many threads requesting an increase from 40 to 200 RPM). Staff have not confirmed it. [Search snippet: https://forums.developer.nvidia.com/t/request-for-nim-api-rate-limit-increase-40-200-rpm/371611]
- An Aliteq article says NVIDIA docs do not print the limit, staff say the maximum appears in the top right of build.nvidia.com, and a moderator said it depends on model, use case and traffic. [Third-party: https://aliteq.com/nvidia-build-free-hosted-models]

Terms:
- Trial Terms text, as quoted in a forum post (secondary): sections 1.2 and 1.4 limit the trial to testing and evaluation, not production; section 4.2 bars distributing Generated Content; section 3.3(iv) cited as allowing NVIDIA to use prompts and outputs to improve products. [Search snippet, secondary quote: https://forums.developer.nvidia.com/t/clarification-on-trial-api-use/334275]
- A separate NVIDIA staff reply says, unless disclosed for a specific API Service, NVIDIA uses User and Generated Content solely to provide the service. [Search snippet: https://forums.developer.nvidia.com/t/api-trial/385386]
- A third-party page says trial terms generally bar storing or using Generated Content after a session, and that production use under a paid subscription is governed by separate terms. [Third-party: https://standardcompute.com/providers/nvidia]
- build.nvidia.com says the trial service is governed by the NVIDIA API Trial Terms of Service, and use of the Nemotron model is governed by the NVIDIA Open Model License Agreement. [Search snippet: https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b/modelcard]
- Licence conflict for Ultra: sources give the NVIDIA Open Model License Agreement (build page) and OpenMDW-1.1 (third-party aggregators). [Third-party: https://haimaker.ai/models/nvidia/nemotron-3-ultra-550b-a55b] Both need checking on the primary card.
- Laguna XS 2.1: OpenMDW-1.1, which permits commercial and non-commercial use. [Verified: https://huggingface.co/poolside/Laguna-XS-2.1-FP8]

Latency:
- NIM.md section 2 and 5 measurements (trivial prompt, 2026-11-20): Laguna 0.66 s total; Ultra thinking off 1.12 s; Ultra thinking-on 25 to 28 s for claim prompts. These are team measurements, not NVIDIA figures.

### Inferences
- The 40 RPM pacing in NIM.md section 4.3 is a planning assumption from forum reports, not a documented contract. Pacing should be driven by observed 429s, not by the 40 figure.
- Pooling several keys to raise effective quota is likely outside the spirit of the trial terms as quoted. Under those terms the approach should not be used for production or for work that counts as production. Confirm with the primary Trial Terms text (see Gaps) before any multi-key design.
- Production options are paid or hosted NIM through NVIDIA AI Enterprise or a hosted NIM provider, per staff, or self-hosted NIM containers under an NVIDIA AI Enterprise licence. Terms and pricing for these were not retrieved. [Hypothesis]

### Gaps
- The primary NVIDIA API Trial Terms of Service text was not retrieved. The claims above come from forum quotes and a third-party page.
- No primary statement of current free-tier RPM. 40 RPM is user-reported only.
- Whether NIM returns `x-ratelimit-*` or `Retry-After` headers is not confirmed. NIM.md section 4.3 says none on success (team-observed).
- Self-hosted NIM container licence, pricing and hardware requirements were not researched.
- Max output and context for NVIDIA-hosted Ultra on NIM are not confirmed by a primary source.

## Prefix and KV caching

### Takeaway
No public NIM caching documentation was retrieved, so the NIM.md section 7 description (automatic server-side prefix cache, visible as `prompt_tokens_details.cached_tokens`) cannot be cross-checked against NVIDIA docs. Treat it as team-observed.

### Cited Findings
- NIM.md section 7 records `usage.prompt_tokens_details.cached_tokens` of 32 and 48 on Laguna, and 0 of 23 on Ultra for a trivial prompt, on 2026-11-20. [Team-observed; NIM.md local file]
- No NVIDIA primary page on NIM prefix caching was fetched.

### Inferences
- Static-first prompt ordering (stable system prompt, volatile data last) is good practice regardless of whether the cache is documented. [Hypothesis]

### Gaps
- Public NVIDIA documentation of caching behaviour, eviction, minimum prefix length and whether it is model-dependent was not found.

## Recommended roles (provisional)

### Takeaway
Use Laguna as the fast executor and compactor, Nemotron 3 Ultra as the planner or verifier where thinking time is acceptable, and GLM-5.3-Flash only as a fallback until its thinking behaviour on NIM is confirmed. All assignments are provisional on live tests.

### Cited Findings
- Laguna: 262K context, 2 to 3 s latency in NIM.md, native tool calling with thinking toggle (see HF card above). [Verified: https://huggingface.co/poolside/Laguna-XS-2.1-FP8]
- Laguna XS 2.1 SWE-bench Verified 70.9%; SWE-bench Multilingual 63.1%; SWE-Bench Pro (public) 47.6%; Terminal-Bench 2.0 37.5% (BF16). Benchmarks run at temperature 1.0, top_k 20, top_p 1, thinking on. Run-to-run variation about 1 to 3 points. [Verified: https://huggingface.co/poolside/Laguna-XS-2.1-FP8]
- Ollama listing reports the same SWE-bench Verified figure. [Search snippet: https://ollama.com/library/laguna-xs-2.1]
- GLM-5.3-Flash: recommended `temperature 1`, `top_p 0.95`. [Verified: https://docs.z.ai/guides/llm/glm-5.3-flash]
- Nemotron 3 Ultra: no tool or agentic benchmarks (BFCL, tau-bench, SWE-bench) were found on a primary card. [Gap]

### Inferences (role mapping)
- Planner/orchestrator: Nemotron 3 Ultra with thinking on, budgeted at 25 to 60 s per step, because it has the largest context and the team's best quality proxy in NIM.md. [Hypothesis]
- Fast executor (tool calls, short steps): Laguna with thinking off, if its NIM tool-call parsing is confirmed. [Hypothesis]
- Summariser/compactor: Laguna, thinking off, with the summary schema checked locally. [Hypothesis]
- Verifier/judge: a different model from the one that produced the output, to avoid self-agreement. Ultra thinking on for hard checks; Nano Omni thinking on as a second opinion if its JSON-enum failures are fixed. [Hypothesis]
- GLM-5.3-Flash: fallback only. Its first-party docs say thinking cannot be disabled, which conflicts with the NIM.md plan. [Hypothesis]

### Gaps
- No benchmark data for tool-use reliability on NIM for any of the five models. Tau-bench, BFCL and similar results were not found on primary sources.

## Live experiments to settle open questions

Run each with the NIM.md probe pattern (1-token key probe, then claim prompt with real schema). Record the HTTP status, finish_reason, `tool_calls` field, and first-token time. Never print keys or raw generated text.

1. Tool-call parsing per model. For each of the five models send `tools` with one function and `tool_choice: "auto"`. Check whether `message.tool_calls` is populated or the call is written into `content`. Pass criterion: parsed `tool_calls` with valid JSON arguments. Use this to fill the support list.
2. Parallel tool calls. Ask for two independent lookups in one turn. Count `tool_calls` entries. Test with and without `parallel_tool_calls: true` (NVIDIA older docs only).
3. Forced and no-tool behaviour. Test `tool_choice` set to "none" and to a named function on each model.
4. Streaming tool-call deltas. Stream a tool call and confirm whether arguments arrive as `delta.tool_calls[index].function.arguments` chunks, and whether `id` and `index` are stable. Check `finish_reason` is `tool_calls`.
5. Thinking controls. For each model compare `chat_template_kwargs.enable_thinking` true and false. Check for `reasoning_content` or `<think>` in output. For GLM, test whether `enable_thinking:false` has any effect, and test `reasoning_effort` low, medium and max.
6. Tools plus schema. Send `tools` and `response_format` json_schema together. Check whether NIM accepts both, and whether the final answer is schema-valid. Then test the two-pass pattern (tools loop, then a schema-only call).
7. Schema keyword set. Re-run NIM.md section 3.1 keyword checks on each model, including `enum`, `anyOf`, `minItems`, `additionalProperties`.
8. Context and output ceilings. Send `max_tokens` at 32,768 and 65,536 to Ultra and GLM, and a near-limit prompt, to confirm the accepted maximum on NIM.
9. Rate-limit behaviour. Burst one key at a rising rate and record the first 429, its body, and any `Retry-After`. Do not use multiple keys to probe limits.
10. Caching. Send the same long system prompt twice and compare `prompt_tokens_details.cached_tokens` per model.
11. Terms check. Before any multi-key design, get the primary NVIDIA API Trial Terms text and the current build.nvidia.com account limit, and have someone with authority decide. Do not treat pooled-key quota as permitted until that is done.

## Source list
- NIM function calling (search snippet only): https://docs.nvidia.com/nim/large-language-models/latest/function-calling.html
- NIM structured generation (empty fetch): https://docs.nvidia.com/nim/large-language-models/latest/structured-generation.html
- NVIDIA developer forum, model limits (fetched): https://forums.developer.nvidia.com/t/model-limits/331075
- NVIDIA forum, trial API use (search snippet): https://forums.developer.nvidia.com/t/clarification-on-trial-api-use/334275
- NVIDIA forum, API trial (search snippet): https://forums.developer.nvidia.com/t/api-trial/385386
- Aliteq, NVIDIA hosted models (search snippet): https://aliteq.com/nvidia-build-free-hosted-models
- Standard Compute, NVIDIA provider page (third-party): https://standardcompute.com/providers/nvidia
- Poolside Laguna XS 2.1 FP8 model card (fetched): https://huggingface.co/poolside/Laguna-XS-2.1-FP8
- Ollama Laguna XS 2.1 (search snippet): https://ollama.com/library/laguna-xs-2.1
- Z.ai GLM-5.3-Flash guide (fetched): https://docs.z.ai/guides/llm/glm-5.3-flash
- Vercel AI Gateway GLM-5.3-Flash (search snippet): https://vercel.com/ai-gateway/models/glm-5.3-flash
- MCP Playground GLM-5.3-Flash (third-party): https://mcpplaygroundonline.com/models/glm-5-3-flash
- CodePick GLM-5.3-Flash guide (third-party): https://codepick.dev/en/guides/glm-5-3-flash-guide/
- iTechGuides GLM-5.3-Flash (third-party): https://www.itechguides.com/how-ai-teams-should-evaluate-glm-5-3-flash/
- build.nvidia.com Nemotron 3 Ultra model card (search snippet only; fetch returned no model text): https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b/modelcard
- Hugging Face NVIDIA-Nemotron-3-Ultra-550B-A55B (fetch returned 404 for the URL tried): https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B
- Haimaker, Nemotron 3 Ultra (third-party): https://haimaker.ai/models/nvidia/nemotron-3-ultra-550b-a55b
- Tech Insider, Nemotron 3 Ultra (third-party): https://tech-insider.org/nvidia-nemotron-3-ultra-open-weight-2026/
- build.nvidia.com Nano Omni model card (search snippet only): https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning/modelcard
- vLLM recipe, Nano Omni (third-party): https://recipes.vllm.ai/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
- LMSYS/SGLang cookbook, Nano Omni (third-party): https://lmsysorg.mintlify.app/cookbook/autoregressive/NVIDIA/Nemotron3-Nano-Omni
- Fireworks, Nano Omni (third-party): https://fireworks.ai/models/fireworks/nvidia-nemotron-3-nano-omni-30b-a3b
- Local contract: `/Users/liamle/Downloads/prophet/NIM.md`

## Gaps (summary)
- Primary NVIDIA pages for Nemotron 3 Ultra and Nemotron 3.5 Super VL were not retrieved (fetch returned empty or 404). Super VL was not researched beyond the NIM.md entry.
- No BFCL, tau-bench or SWE-bench-with-agent results for Nemotron models were found on a primary source.
- NIM structured-generation and caching docs were not retrieved.
- Primary NVIDIA API Trial Terms text not retrieved; terms claims rely on forum quotes and a third-party page.
- Self-hosted NIM licence and pricing not researched.
- Laguna and Nano Omni NIM-side tool and thinking behaviour is unconfirmed; the HF and vendor docs describe self-hosted vLLM.
