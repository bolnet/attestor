# Voiceover script — setup-05-local-bedrock-embeddings.webm

**Video:** `docs/demo/setup-05-local-bedrock-embeddings.webm`
**Duration:** ~2m 20s (140s)
**Variant:** 5 of 8 — Local backend, **AWS Bedrock** embeddings (vs local sentence-transformers)
**What changes from variant 01:** Q4 picks the cloud-native embeddings group and selects Bedrock on the sub-question. SQLite, ChromaDB, and NetworkX still run locally — only the embedding calls go to Bedrock. Everything else is defaults.
**Tone:** calm, confident. One beat slower than conversational.

---

## Open — Claude Code header (0:00 – 0:08)

> Same install command. Same local backend. One knob turned.
>
> This time the embeddings don't run on your laptop — they run inside your AWS account, on Bedrock.

## "install attestor" prompt typed (0:08 – 0:18)

> Three words. *Install agent memory.* The wizard runs the same seven questions. The first three we take as defaults.

## Q1 — scope, default Global (0:18 – 0:28)

> Global scope. Default, same as variant one.

## Q2 — store path, default (0:28 – 0:38)

> `~/.attestor`. Default, same as variant one.

## Q3 — backend, Local (0:38 – 0:50)

> Local backend. SQLite, ChromaDB, NetworkX — all embedded, all on this machine. Same as variant one. The only thing we're changing today is where the embeddings come from.

## Q4 — embeddings, AWS Bedrock (0:50 – 1:35)

> Here's the change. Embeddings provider.
>
> Local sentence-transformers is the default — 90 megabytes, offline, free. OpenAI is option two — strongest quality, but your text leaves your network and lands in OpenAI's. For regulated environments, neither of those is acceptable.
>
> Option three: *Bedrock, Vertex, Azure.* Cloud-native embeddings that stay inside your cloud perimeter. Two arrow-key downs. Enter.
>
> Sub-question: which provider. Default is Bedrock — AWS.
>
> Why Bedrock specifically. First, data residency: your text never leaves your AWS account. The embedding call goes to a Bedrock endpoint in the region you already operate in, under the same VPC, the same CloudTrail, the same compliance boundary as the rest of your infrastructure.
>
> Second, identity: no separate API key to provision, rotate, or leak. The Attestor process picks up your existing IAM credentials — instance profile, IRSA, SSO, whatever you already use — and calls Bedrock with them. One fewer secret in your environment.
>
> Third, compliance: for HIPAA, FedRAMP, SOC 2 environments where shipping text to a third-party embedding API is off the table, Bedrock keeps the entire path inside a contract you already have with AWS.
>
> The trade-off is honest. You need AWS credentials configured before `attestor doctor` runs. You pick a region. And per-call latency is a network hop higher than local — tens of milliseconds instead of sub-millisecond. For background ingestion that's invisible; for hot-path recall the embedding is already cached on the memory at write time, so read latency is unaffected.
>
> Same answer also exposes Vertex AI on GCP and Azure OpenAI Service — identical reasoning for teams already standardized on those clouds. Pick the one whose IAM you already trust.
>
> Enter to accept Bedrock.

## Q5 — hooks, all three (1:35 – 1:55)

> Hooks. Session-start, post-tool-use, stop. All three recommended. Default, same as variant one.

## Q6 — namespace, default (1:55 – 2:02)

> Namespace `user`. Default, same as variant one.

## Q7 — token budget, default (2:02 – 2:10)

> Ten thousand tokens. Default, same as variant one.

## Install + doctor + MCP merge (2:10 – 2:20)

> Claude runs `uv tool install attestor`, then `attestor doctor`. Doctor now has one extra check — it calls Bedrock with your IAM credentials to prove the embedding path works end-to-end before wiring the MCP server into `~/.claude/.mcp.json`.
>
> If the IAM role can't reach Bedrock, doctor fails loudly here, not later during a live recall. That's the whole point of running it up front.

---

## Recording notes

- Make sure AWS credentials resolve in the shell VHS spawns — either `aws configure` has been run on the host, or `AWS_PROFILE` is exported, or an IAM instance profile is attached. Otherwise `attestor doctor` will fail inside the recording and the last ten seconds look broken.
- The Q4 sub-question (Bedrock / Vertex / Azure picker) is the one new interaction relative to variants 01–04. Hold the voiceover for the extra beat while it renders — don't pre-narrate "user picks Bedrock" before the sub-card appears.
- Spend the screen time on Q4. Q1–Q3 and Q5–Q7 are identical to variant 01; the viewer just needs to see the selector move past them. Don't re-explain scope, store path, or hooks at length — defer to variant 01 and keep the pace up.
- Voiceover cadence should match variants 01 and 02 so the series feels consistent. One beat behind the keystroke, not ahead of it.
- If you re-record and AWS credentials aren't available in the VHS shell, fall back to variant 01 or cut the recording at the end of Q4 and stitch a static "doctor output" frame rather than showing a failed call.
