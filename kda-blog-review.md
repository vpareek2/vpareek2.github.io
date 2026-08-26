# Autoresearch on Kimi Delta Attention

> What a closed-loop AI optimization process looks like in practice.

This document collects the article copy and reader-facing interactive text in page order. Repetitive chart-point labels are collected in the appendix.

---

## Exact training throughput

### From 833 to 44,942 tokens/s

**Legend:** Kept · Discarded

**Milestones**

1. **833 tokens/s: Initial eager PyTorch implementation.** Python-loop baseline: 38.2 seconds per training update.
2. **7,394 tokens/s: Project CUDA baseline.** First practical fully owned CUDA training path.
3. **28,325 tokens/s: Tensor-core recurrence.** A unified WMMA scan moved recurrence onto tensor cores.
4. **36,185 tokens/s: Flattened backward.** A flatter pair grid removed 18 kernel launches.
5. **42,237 tokens/s: Compact BF16 dataflow.** BF16 publication removed large FP32 surfaces.
6. **44,942 tokens/s: Release confirmation.** Three-run median.

Every dot is a measured trainer candidate and every run appears in the log. Six brighter milestones trace the retained story; select any point to jump to its entry. Historical runs span several matched measurement blocks. The final point is a fresh three-run confirmation of the retained exact source.

---

## A one-day experiment

In 24 hours, an autoresearch loop improved the training throughput of a six-layer Kimi Delta Attention (KDA) development model on an NVIDIA GB10 from 833 to 44,942 tokens per second. The final exact implementation exceeded the 43,937 tokens per second reached by the [Flash Linear Attention (FLA) library](https://github.com/fla-org/flash-linear-attention) under the same model, workload, and device configuration.

This result is deliberately narrow. The model architecture, sequence length, precision, optimizer, data ordering, and KDA math were held constant. The agent was only allowed to change how that computation was implemented on the hardware.

What interested me was not only the final number, but how the system got there.

### Why optimize KDA?

I first became interested in KDA when the [Kimi Linear paper](https://arxiv.org/abs/2510.26692) was released. A personal architecture project I was working on explored interleaving linear attention with sliding-window attention, and KDA looked like a natural candidate. The project was written in JAX, however, so the PyTorch-oriented kernels released through FLA were not something I could simply drop in.

That inconvenience became the experiment: if I fixed one hardware target, one workload, and strict numerical constraints, how far could an autoresearch loop push a correct but impractical implementation in one day?

### KDA in one minute

Standard attention keeps every earlier key and value available. KDA compresses the past into a fixed-size matrix, a form of working memory. Each token reads what that memory currently predicts and writes only the correction. A learned gate lets different memory channels forget at different rates. The result is a memory that continually edits itself as the model reads.

#### 1. Write

- **Incoming token:** Key `ALPHA`, value `4`.
- Turn the token into an address and some content.
- **Memory action:** Add association.
- Address `ALPHA` now points toward `4`.
- The state grows no larger when the sequence does.
- Unlike standard attention, KDA does not retain every earlier key and value. It continually updates one compact working memory.

#### 2. Collide

- **Incoming token:** Key `ALPHA′`, value `7`.
- A similar address arrives with different content.
- **Memory action:** Read current estimate.
- Similar addresses overlap: old + new produces a mixed estimate.
- A fixed state inevitably creates interference between associations.
- A finite memory cannot give every token its own private slot. Similar keys overlap, so blindly adding another value would preserve the error.

#### 3. Correct

- **Incoming token:** Target value `7`.
- The target is compared with what memory already predicts.
- **Memory action:** Erase error; write difference.
- Update only `target − estimate` rather than adding another full copy.
- This prediction error is the “delta” in the delta rule.
- KDA reads its current prediction, measures the error, and writes the correction. The memory learns while the model reads.

#### 4. Forget

- **Incoming token:** Key `ALPHA′`, value `7`.
- Before the next correction, the old state is selectively decayed.
- **Memory action:** Decay each channel.
- Each memory channel gets its own retention rate, preserving useful structure while fading stale content.
- KDA makes forgetting fine-grained instead of using one rate for the whole head.
- Some channels remember; others reset quickly. That fine-grained control is KDA’s main extension to Gated DeltaNet.

**The systems problem:** Token 1 → State 1 → Token 2 → State 2 → …

Each state appears to depend on the one before it, which is the opposite of the wide parallel work GPUs prefer.

For the full technical explanation: [DeltaNet I: the model](https://sustcsonglin.github.io/blog/2024/deltanet-1/), [DeltaNet II: the algorithm](https://sustcsonglin.github.io/blog/2024/deltanet-2/), and the [Kimi Linear paper](https://arxiv.org/abs/2510.26692).

The recurrence is easy to express in PyTorch, but efficient training requires parallelizing it across tokens without dropping any part of the forward computation or backward pass. The gap between a correct implementation and a hardware-efficient one was the target of this experiment.

The algorithmic escape hatch is already known. [DeltaNet’s chunkwise formulation](https://sustcsonglin.github.io/blog/2024/deltanet-2/) groups tokens into blocks: memory still moves sequentially between blocks, while most of the work inside each block becomes parallel matrix multiplication. The Kimi Linear paper derives the corresponding chunkwise form for KDA.

This project does not invent that algorithm. Kimi Linear supplied the mechanism and formulation, while FLA supplied the performance target. The experiment was whether an agent could turn those ideas into an exact, hardware-specific training path for the GB10 and optimize it far enough to compete.

---

## Designing the autoresearch loop

Autoresearch was not a single prompt. It was a closed experimental system: the model proposed and implemented changes, frozen gates determined what counted, and a durable record carried evidence from one attempt to the next.

The agent owned the fast inner loop. I owned the objective and periodically reopened the global profile when a locally productive search stopped attacking the largest remaining bottleneck.

### One experiment at a time

**Codex + Prime-Agent · GPT-5.6-Terra / high**

1. **Profile: Read the global profile.** Start from measured end-to-end cost and choose one bottleneck large enough to matter.
2. **Hypothesize: Name one mechanism.** Explain the bottleneck and predict a measurable effect before changing the implementation.
3. **Implement: Build one candidate.** Change one primary scheduling, layout, fusion, or dataflow idea in Python or CUDA.
4. **Verify: Try to falsify it.** Check the forward pass and random-upstream gradients against an independent PyTorch oracle. A numerical mismatch rejects the candidate; a crash or missing measurement marks the run invalid.
5. **Benchmark: Measure the real workload.** Only exact candidates reach the matched, production-shaped six-layer trainer benchmark.
6. **Keep or revert: Record the result.** A material exact win becomes the baseline. Every other outcome still enters the experiment ledger.

**Frozen outside the loop:** Exact KDA · full gradients · matched workload · independent oracle.

I ran the model through a mix of the Codex and Prime-Agent harnesses, often using `/goal` to keep the longer objective explicit. The exactness gates, matched benchmark, and append-only ledger made the work recoverable and auditable. FLA remained a performance comparator; an independent PyTorch path was the numerical reference.

The agent was extremely strong at executing and evaluating a concrete experiment. My highest-leverage role was noticing when a locally productive search had stopped serving the global objective, then forcing the loop back to the end-to-end profile and changing course.

---

## Reading the campaign as logs

The final implementation was not the result of one large breakthrough. It emerged from hundreds of profile, correctness, and benchmark events: some cumulative, many negative, and a few that changed the direction of the search.

The logs below condense the record. Attempt identifiers and measurements come from the campaign ledger; the narration is shortened for readability. Rejected and out-of-contract results remain visible because they were part of deciding what the headline could honestly mean.

### Condensed campaign trace

`GB10 / 24 HOURS / EXACT TRAINING`

```text
$ kda-autoresearch --device GB10 --budget 24h
[CONTRACT] exact KDA · full gradients · matched six-layer trainer
[TRACE] 366 ledger events · 86 measured trainer candidates

-- PHASE 1 / ESTABLISH THE BASELINE --

[BASELINE] Initial eager implementation
  [PROFILE]   Python recurrence: 38.2 seconds per training update
  [REFERENCE] Full KDA forward and backward in eager PyTorch
  [BENCH]     833 TOKENS/S

[AGENT] continued improving the Python path; a CUDA rewrite remained deferred
[HUMAN REDIRECT] Stop extending the Python path. Own the complete CUDA training implementation.

-- PHASE 2 / MOVE THE PATH INTO CUDA --

[KEEP 14] Parallel backward history · 1,978 TOKENS/S
  Backward state-history work was distributed instead of replayed serially.
[KEEP 15] Parallel forward history · 2,866 TOKENS/S
  The corresponding forward history moved into the project-owned parallel path.
[KEEP 18] Value-tiled reverse · 6,689 TOKENS/S
  The reverse recurrence was split across value tiles to expose more independent GPU work.

[MILESTONE 19] First practical project CUDA baseline
  [BOTTLENECK] Python dispatch and serial history dominated the update
  [CHANGE]     Complete project-owned CUDA forward and backward
  [VERIFY]     Independent gradients · runtime ownership · no fallback
  [BENCH]      833 → 7,394 TOKENS/S

-- PHASE 3 / PARALLELIZE THE RECURRENCE --

[KEEP 23] Bounded convolution dependencies · 10,957 TOKENS/S
  Dependency structure was rewritten so chunk-local work could proceed in parallel.
[KEEP 36] Row-parallel pair VJP · 21,847 TOKENS/S
  Independent rows of the pairwise vector-Jacobian product received separate owners.
[REJECT 56] Hybrid WMMA scan · BELOW GATE
  A partial tensor-core conversion was correct, but its measured gain was too small to advance.

[MILESTONE 65] Tensor-core recurrence
  [BOTTLENECK] Recurrent scans and pair transforms remained on the critical path
  [CHANGE]     Chunkwise transforms plus a unified forward/backward WMMA scan
  [VERIFY]     Random-upstream gradients · production-shape gate
  [RUNNING BEST] 7,394 → 28,325 TOKENS/S

-- PHASE 4 / FLATTEN THE BACKWARD PASS --

[KEEP 84] Build-pair WMMA · 31,747 TOKENS/S
  Pair construction and its tensor-core consumers were brought into one retained path.
[REJECT 123] BF16 chunk-state history · 29,354 TOKENS/S
  Lower-precision history reduced storage but introduced enough extra work to regress the trainer.
[KEEP 161] Fast math and generic fallback · 35,521 TOKENS/S
  A guarded fast path accelerated the production shape while preserving the exact generic route.

[MILESTONE 168] Flattened parallel backward
  [BOTTLENECK] Ten independent triangular pair families launched separately
  [CHANGE]     Flatten pair ownership into one broader CUDA grid
  [PROFILE]    Operator launch count: 185 → 167
  [RUNNING BEST] 36,185 TOKENS/S

-- PHASE 5 / SHRINK THE DATAFLOW --

[KEEP 204] Forward group checkpoints · 38,803 TOKENS/S
  Only the group-level state needed by backward remained materialized.
[KEEP 217] Correct retained WY layout · 40,347 TOKENS/S
  Retained factors were laid out for their consumers without changing the WY computation.
[KEEP 256] BF16 forward WY products · 42,109 TOKENS/S
  Selected forward products crossed the producer-consumer boundary directly in BF16.

[MILESTONE 266] Compact BF16 dataflow
  [BOTTLENECK] Large FP32 U/W surfaces were written, packed, and reread
  [CHANGE]     Publish U to compact scratch and W directly to retained storage
  [VERIFY]     Seven random-dO gradients bitwise equal · four sanitizers clean
  [DIRECT LEVEL 2] 40,105 → 42,237 TOKENS/S

-- PHASE 6 / AUDIT THE LAST MILE --

[DISCARD 271] Forward CUDA Graph · 42,262 TOKENS/S
  A locally positive graph result did not establish a durable retained improvement.
[REJECT 274] Stacked CUDA Graphs · 20,883 TOKENS/S
  Composing individually plausible captures produced a severe end-to-end regression.
[KEEP 342] Group-major retained data · 43,840 TOKENS/S
  Producer layouts were aligned with the backward consumers that reused them.
[KEEP 345] Fused RMSNorm and output gate · 44,542 TOKENS/S
  An exact vertical fusion removed a layer boundary and passed a three-by-three matched trainer comparison.

[OUT OF CONTRACT] Local-path surrogate exceeded 45.5K by truncating KDA backpropagation
[OUT OF CONTRACT] Value-only surrogate reached 50,541.5 by omitting trainable gradient paths
[HUMAN AUDIT] A faster number does not count when the computation has changed.

[NEUTRAL 372] Exact BF16 weight cache · +4 TOKENS/S
  Bitwise exact, but the apparent microbenchmark gain disappeared in the matched trainer.
[REJECT 373] Four-CTA clustered VJP · +3.85% TIME
  Fewer launches lost to idle cluster residency and barriers; the exact layer became slower.

[RELEASE] Exact matched confirmation
  [PROJECT RUNS] 45,058 · 44,942 · 44,842 TOKENS/S
  [FLA RUNS]     43,958 · 43,898 · 43,937 TOKENS/S
  [CONFIRMED MEDIANS] 44,942 vs. 43,937 · +2.287%
  [STRONGEST OBSERVED RUN] 45,058 TOKENS/S

[COMPLETE] exact source retained · evidence saved · claim bounded to this workload and GB10
```

---

## Twenty minutes of actual training

The seven-step benchmark isolates steady-state throughput, but I also wanted a more tangible test: given the same twenty minutes, how many complete training updates would each implementation actually finish?

I ran three fresh copies of the same six-layer training job, changing only the KDA backend. Each run used two unscored warm-up updates followed by twenty minutes of measured training. Eager PyTorch completed **12 updates**, FLA completed **1,617**, and the project CUDA backend completed **1,641**. That is 24 more updates than FLA, a **1.48% lead**.

### 20 minutes · same model · same GB10

**Eager PyTorch: 12 updates**

- 393,216 measured tokens
- 321 tokens/s sustained
- Final displayed smoothed loss: 9.092

**FLA: 1,617 updates**

- 52,985,856 measured tokens
- 44,143 tokens/s sustained
- Final displayed smoothed loss: 3.850

**Project CUDA: 1,641 updates**

- 53,772,288 measured tokens
- 44,783 tokens/s sustained
- Final displayed smoothed loss: 3.811

<!--
Visualization direction: three aligned small-multiple charts, side by side on
wide screens and stacked on narrow screens. Use measured training time (0–20
minutes) on the shared x-axis and the same smoothed-loss domain on the y-axis.
Label each endpoint with completed updates and final loss. Make the project
panel slightly brighter and add a +24 updates / +1.48% badge relative to FLA.
Do not use step number as the primary x-axis: the learning-rate schedule is
wall-time based, so equal step numbers occur at different schedule positions.
Allow a steps/time toggle only as a secondary exploratory control.
-->

The project run also ended with the lowest displayed loss. I treat that only as a consistency check, not as a quality result: this was one short seed, the project completed 24 additional updates, and the remaining difference is not separable from ordinary BF16 and reduction-order effects without repeated runs. The defensible win is simpler: it completed the most exact training work under the fixed time budget. Neither optimized run fell back to another backend. Spot checks showed no active NVIDIA slowdown flags, although this direct runner did not capture continuous thermal telemetry.

Full traces: [Eager PyTorch](https://wandb.ai/veerpareek12/nanochat/runs/zxbrirrf) · [FLA](https://wandb.ai/veerpareek12/nanochat/runs/5k9ebdl5) · [Project CUDA](https://wandb.ai/veerpareek12/nanochat/runs/cn0gwiy4)

---

## What the experiment showed

This experiment is not evidence that an autoresearch loop can replace a library like FLA, CUDA experts, or diligent systems work. The result is purposefully narrow: one algorithm, one chip, and one pinned comparison. On that setup, after 24 hours of optimization, the project implementation reached a confirmed median of 44,942 tokens per second, 2.287% above FLA. It does raise a more interesting question: how should future optimization systems divide the work between human judgment and agent execution?

My kernel-development background is limited. I took [ECE 408: Applied Parallel Programming](https://ece.illinois.edu/academics/courses/ece408-120238) at UIUC and have worked through some [GPU MODE](https://github.com/gpu-mode) lectures, which gave me enough context to understand the mechanism and audit a technical discussion. I could not have written this implementation from memory. Doing it conventionally would have taken me weeks, assuming I got there at all.

That does not mean the experiment was hands-off. The agent was highly effective when given a concrete bottleneck and a reliable way to test its work. It was less reliable at deciding when a productive local search had stopped addressing the global goal. My role was to audit its conclusions, reopen the end-to-end profile, and redirect the search when it became attached to increasingly small improvements.

The final hours made that division of labor especially clear. Several surrogate variants produced numbers above the exact implementation, and one crossed 50,000 tokens per second by omitting trainable gradient paths. Those results were interesting as diagnostics, but they were not KDA training. Without a frozen oracle, full-gradient checks, and a benchmark that failed closed, it would have been easy to publish the larger, incorrect number.

This also changed how I think about autograd. I do not expect it to disappear from research code: it remains a remarkably useful way to express and revise a model. What changes is the cost of leaving it. Once an architecture and workload stabilize, an agent can help migrate the hot path into lower-level primitives much sooner than I would have attempted by hand. The difficult skill shifts from personally writing every kernel to specifying the computation, building tests that can falsify a candidate, and recognizing when the search is optimizing the wrong thing.

The most useful result, then, is not simply that the final kernel edged past FLA. It is that the distance between understanding a mechanism and obtaining a competitive, hardware-specific training implementation became much shorter.

---

## Appendix: complete throughput-chart labels

Historical values are comparable only within their declared matched measurement blocks. “Kept” means retained exact improvement; “discarded” means measured but not retained.

| Attempt | Status | Tokens/s | Label |
|---:|---|---:|---|
| 0 | Milestone | 833 | Initial eager implementation |
| 14 | Kept | 1,978 | Parallel backward history |
| 15 | Kept | 2,866 | Parallel forward history |
| 18 | Kept | 6,689 | Value-tiled reverse |
| 19 | Milestone | 7,394 | First practical project CUDA baseline |
| 23 | Kept | 10,957 | Convolution bounded dependencies |
| 24 | Kept | 12,977 | Factored convolution gradient |
| 26 | Kept | 14,800 | FP32 C64 BMM scan |
| 27 | Discarded | 14,981 | BF16 WMMA C64 scan |
| 28 | Kept | 15,895 | Chunk-boundary recompute |
| 34 | Kept | 19,009 | Local raw-gate gradient |
| 36 | Kept | 21,847 | Row-parallel pair VJP |
| 38 | Kept | 23,851 | Parallel exact beta gradient |
| 43 | Kept | 24,310 | Fused stable transforms |
| 45 | Kept | 25,420 | Key-major parameter reduction |
| 46 | Kept | 26,258 | Boundary key block |
| 47 | Discarded | 26,685 | Finalize row block |
| 51 | Kept | 27,043 | Bounded stable pair VJP |
| 53 | Kept | 27,671 | Chunk partial finalization |
| 65 | Milestone | 28,325 | Tensor-core recurrence |
| 67 | Discarded | 28,663 | Pair pack 256 |
| 68 | Kept | 28,784 | Eight-warp forward |
| 70 | Kept | 29,699 | Eight-warp group recompute |
| 72 | Kept | 29,868 | Value-tile group state |
| 77 | Kept | 29,921 | Reverse transfer fusion |
| 83 | Kept | 30,637 | Pair WMMA VJP |
| 84 | Kept | 31,747 | Build pair WMMA |
| 86 | Discarded | 31,981 | Persistent build solve |
| 89 | Discarded | 31,856 | BF16 inter-chunk state |
| 91 | Kept | 32,914 | Persistent reverse group |
| 93 | Discarded | 33,226 | Fused group producer |
| 95 | Discarded | 33,348 | Post-reverse WMMA VJP |
| 97 | Discarded | 33,236 | Two-CTA post-reverse VJP |
| 100 | Kept | 33,601 | Colored pair VJP |
| 110 | Discarded | 34,210 | Backward group-major layout |
| 111 | Discarded | 34,029 | Group-major producer |
| 118 | Discarded | 33,689 | Forward tile prefetch |
| 123 | Discarded | 29,354 | BF16 chunk-state history |
| 125 | Discarded | 34,413 | Async scan operands |
| 127 | Kept | 34,494 | Preprocess q-gamma |
| 133 | Discarded | 34,926 | Two-group short-path guard |
| 134 | Discarded | 34,549 | Boundary decay cache |
| 154 | Discarded | 34,468 | Fused state dot |
| 156 | Discarded | 34,672 | Fused BF16 reverse products |
| 161 | Kept | 35,521 | Fast math and generic fallback |
| 162 | Discarded | 35,984 | Four-warp VJP |
| 165 | Discarded | 35,743 | Fused U/W pack |
| 168 | Milestone | 36,185 | Flattened parallel backward |
| 173 | Discarded | 35,981 | Four-CTA VJP |
| 175 | Kept | 36,719.5 | Tiled convolution backward |
| 189 | Discarded | 37,198 | Register dP consumer |
| 190 | Discarded | 37,519 | Register dQ consumer |
| 194 | Discarded | 37,701 | Register state products |
| 201 | Discarded | 38,052 | Direct WMMA dH scan |
| 204 | Kept | 38,803 | Forward group checkpoints |
| 211 | Discarded | 39,560 | GB10 register forward state |
| 213 | Kept | 40,076 | Retained forward WY factors |
| 217 | Kept | 40,347 | Correct retained WY layout |
| 221 | Discarded | 39,784 | Persistent build solve |
| 222 | Discarded | 40,632 | Pipelined key products |
| 227 | Discarded | 40,707 | Split fused boundary dH |
| 231 | Kept | 40,834 | Hidden dA tail |
| 243 | Discarded | 41,565 | Fused retention scan pack |
| 244 | Discarded | 41,523 | Fused reverse base gradient |
| 245 | Kept | 41,657 | Retained warp normalization |
| 255 | Kept | 41,922 | Swapped retained P for prefix |
| 256 | Kept | 42,109 | BF16 forward WY products |
| 265 | Discarded | 41,856 | Direct publication register VJP |
| 266 | Milestone | 42,237 | Compact BF16 dataflow |
| 268 | Discarded | 42,199 | BF16 U/W register VJP |
| 270 | Discarded | 41,975 | Fused qg/kg producer |
| 271 | Discarded | 42,262 | Forward CUDA Graph |
| 272 | Discarded | 40,023 | Reverse-group CUDA Graph |
| 274 | Discarded | 20,883 | Stacked CUDA Graphs |
| 279 | Discarded | 41,249 | Interleaved group pack |
| 282 | Discarded | 41,530 | Restored-k interleaved stack |
| 283 | Discarded | 41,132 | Fused preprocess/build stack |
| 285 | Discarded | 42,064 | BF16 group-U rebuild |
| 289 | Discarded | 42,721 | Backward zero-fill producers |
| 292 | Discarded | 41,680 | Optimized host wrapper |
| 308 | Discarded | 43,145 | Preprocess and convolution stack |
| 321 | Discarded | 43,260 | FP16 normalized forward scratch |
| 325 | Discarded | 43,572 | Fused best stack |
| 335 | Discarded | 43,572 | Compact retained q/k/P |
| 342 | Kept | 43,840 | Group-major retained data |
| 366 | Milestone | 44,942 | Exact release confirmation |
