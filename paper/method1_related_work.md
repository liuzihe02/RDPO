# Related Work: Method 1 (Critique DPO) in the Literature

## The Objective

Method 1 expands the DPO loss with an additional NLL term on the oracle/verifier's chain-of-thought rationale:

$$\mathcal{L}_{\theta}(\mathcal{X}_{R}) = - \mathbb{E}_{\mathcal{X}_{R}} \left[ \log \sigma \left( \hat{r}(x, y_{w}) - \hat{r}(x, y_{l}) \right) + \log p_{\theta}(r \mid c : y_{w}, y_{l}) \right]$$

where $r$ is the oracle's CoT reasoning for why $y_w \succ y_l$, and $c$ are the verifier's instructions. The base model $p_\theta$ is trained jointly to (1) prefer better answers via the DPO term, and (2) reproduce the verifier's reasoning via the NLL term.

Three features define this combination:

| Feature | Description |
|---|---|
| **A** | DPO loss on answer preferences $(y_w, y_l)$ |
| **B** | NLL on the *verifier/oracle's* CoT critique $r$ |
| **C** | Applied to the **base/policy model**, not a separate judge |

---

## How the Existing Literature Maps On

No single published paper matches all three features simultaneously. The field occupies adjacent positions.

### Gets A+C but not B — DPO + NLL on the model's own reasoning

**Iterative Reasoning Preference Optimization** (Pang et al., 2024) is structurally the closest paper. It uses:

$$\mathcal{L} = -\log \sigma\!\left(\beta \log \frac{M_\theta(c^w, y^w|x)}{M_i(c^w, y^w|x)} - \beta \log \frac{M_\theta(c^l, y^l|x)}{M_i(c^l, y^l|x)}\right) - \alpha \frac{\log M_\theta(c^w, y^w|x)}{|c^w| + |y^w|}$$

The DPO + NLL skeleton is identical to Method 1. The key difference: $c^w$ is the **model's own chain-of-thought for solving the problem** — the winning reasoning trace — not a verifier's critique of the *comparison* between two responses. The NLL term reinforces self-generated correct reasoning; Method 1's NLL term injects an external perspective on *why* one response is better.

**AIPO** (Shen et al., 2024) has the same structure: DPO + NLL on the winning response $y_w$, again training on the model's own output rather than a verifier's rationale.

### Gets B+A but not C — DPO + NLL on verifier rationale, but for a judge

**Direct Judgement Preference Optimization** (Wang et al., 2024) uses the same objective form as Method 1 including the rationale NLL term, but the goal is to train a *judge/evaluator* model rather than the policy that generates answers. The model being optimised is a standalone judge, not the base model.

**GenRM-STaR-DPO** (Mahan et al., 2024) similarly applies DPO to (correct rationale + judgement) vs (incorrect rationale + judgement) pairs, but again for a dedicated judge model.

### Gets B+C but not A — NLL on critique for the base model, no DPO term

**Critique Fine-Tuning (CFT)** (Wang et al., 2025) fine-tunes the base model by maximising the likelihood of annotated critique reasoning $P(c \mid [x; y])$. This is exactly the NLL term in Method 1, isolated. The paper reports 4–10% improvement over SFT on six math benchmarks. Adding the DPO preference term from Method 1 on top of CFT is a direct extension.

---

## Newly Published Papers (2024–2026) with Close Overlap

### Critique-GRPO (Zhang et al., 2025 — arXiv:2506.03106)

The closest newly published paper to Method 1 in spirit. It integrates natural-language critiques and numerical rewards jointly in a GRPO-based RL objective for the **base model**:

- Trains the same model to generate correct responses **and** generate critiques of its own initial responses
- Both NL critique quality and numerical answer correctness feed into the training signal simultaneously

The structure maps almost exactly onto Method 1 with two differences:

| | Method 1 | Critique-GRPO |
|---|---|---|
| Algorithm | DPO (offline) | GRPO (online RL) |
| Critique source | External oracle/verifier | Self-generated |
| Critique use | NLL term in DPO loss | RL reward shaping |

If Method 1 is "offline, DPO-based Critique-GRPO with an external verifier", these two papers converge on the same intuition from different directions. Critique-GRPO achieves +4.4% on Qwen2.5-7B-Base and +16.7% pass@1 on AIME 2024.

### SuperCorrect (Yang et al., 2024 — arXiv:2410.09008)

Uses cross-model collaborative DPO: a teacher model generates error-correction traces for the student base model, which is then trained via DPO on those traces. The teacher's correction is functionally analogous to $r$ in Method 1 — an external model's reasoning about what went wrong and how to fix it. The difference is that SuperCorrect's DPO term compares full correction traces rather than separating the answer preference signal (A) from the critique NLL (B).

### RL Tango (Zha et al., 2025 — arXiv:2505.15034)

Jointly trains a generator and a generative process-level verifier using interleaved RL, with the verifier rewarded solely on outcome correctness. The verifier's CoT naturally emerges as a training signal for the generator without requiring pre-labelled critique data. This validates the core premise of Method 1 — that verifier reasoning is a useful learning signal for the generator — but via co-training rather than a combined loss.

---

## Summary Table

| Paper | A: DPO on answers | B: NLL on verifier CoT | C: Base model |
|---|:---:|:---:|:---:|
| **Method 1 (this paper)** | ✓ | ✓ | ✓ |
| IRPO (Pang et al., 2024) | ✓ | NLL on own CoT | ✓ |
| AIPO (Shen et al., 2024) | ✓ | NLL on $y_w$ | ✓ |
| DJPO (Wang et al., 2024) | ✓ | ✓ | ✗ (judge) |
| GenRM-STaR-DPO (Mahan et al., 2024) | ✓ | ✓ | ✗ (judge) |
| CFT (Wang et al., 2025) | ✗ | ✓ | ✓ |
| **Critique-GRPO** (Zhang et al., 2025) | GRPO not DPO | self-critique | ✓ |
| SuperCorrect (Yang et al., 2024) | on traces | teacher correction | ✓ |
| RL Tango (Zha et al., 2025) | via RL | verifier CoT | co-training |

---

## Positioning

Method 1 sits at the intersection of three lines of work that have so far only been combined in pairs:

1. **IRPO's DPO+NLL structure** — the right loss form, wrong NLL target
2. **DJPO/GenRM's use of verifier rationale as the NLL target** — the right data, wrong model being trained
3. **CFT's base-model orientation** — the right model, missing the DPO preference term

The most natural framing for a related work section is: Method 1 can be understood as IRPO (Pang et al.) where the NLL term is placed on the *verifier's critique* rather than the model's own reasoning — or equivalently, as CFT (Wang et al.) augmented with a DPO preference term. Critique-GRPO (Zhang et al., 2025) arrives at a closely analogous idea from the RL direction, providing independent empirical validation that this combination of critique reasoning and preference optimisation for the base model is a productive direction.

# Method 1 (Critique DPO) — Related Work Ranking

## The Objective

$$\mathcal{L}_{\theta}(\mathcal{X}_{R}) = - \mathbb{E}_{\mathcal{X}_{R}} \left[ \log \sigma \!\left( \hat{r}(x, y_{w}) - \hat{r}(x, y_{l}) \right) + \log p_{\theta}(r \mid c : y_{w}, y_{l}) \right]$$

Three defining features:

| Label | Feature |
|---|---|
| **A** | DPO loss on answer preferences $(y_w, y_l)$ |
| **B** | NLL on the **oracle/verifier's CoT rationale** $r$ (why $y_w \succ y_l$) |
| **C** | Applied to the **base/policy model** — not a separate judge or reward model |

No published paper has all three. Below is a ranked list from most to least relevant.

---

## Tier 1 — Structurally Identical, One Component Differs

These papers match two of the three features exactly; they are the most important to cite and distinguish.

### 1. Iterative Reasoning Preference Optimization (Pang et al., 2024) ★★★★★
*Already referenced · arXiv:2404.19733*

**Has A + C. Differs on B.**

The loss is identical in form — DPO term + NLL term — but the NLL is on the model's **own** winning chain-of-thought $c^w$ for solving the task, not a verifier's critique of the comparison between $y_w$ and $y_l$. This is the single most structurally similar paper. Method 1 can be framed precisely as: *IRPO where the NLL target is replaced with the oracle's preference rationale $r$.*

---

### 2. Direct Judgement Preference Optimization (Wang et al., 2024) ★★★★★
*Already referenced · arXiv:2409.14664*

**Has A + B. Differs on C.**

Uses the same DPO + NLL-on-rationale objective form as Method 1, including using a verifier's reasoning trace as the NLL target. The critical difference: it trains a **judge model**, not the policy model that generates answers. Method 1 is DJPO applied to the base model.

---

### 3. Critique Fine-Tuning / CFT (Wang et al., 2025) ★★★★★
*Already referenced · arXiv:2501.xxxxx*

**Has B + C. Differs on A.**

Fine-tunes the base model with NLL on annotated critique reasoning $P(c \mid [x; y])$ — this is exactly the second term in Method 1, in isolation. Adding the DPO preference term (feature A) directly yields Method 1. CFT alone achieves 4–10% improvement on math benchmarks purely from the critique NLL signal.

---

### 4. **Critique-GRPO** (Zhang et al., 2025) ★★★★★
*NEW · arXiv:2506.03106*

**Has A + C. Differs on B (self-critique, not external) and algorithm (GRPO not DPO).**

Integrates natural-language critiques and numerical rewards jointly in GRPO for the base model — the most conceptually similar new paper to Method 1. The model learns simultaneously from response preferences and NL critique reasoning. Key differences from Method 1:

| | Method 1 | Critique-GRPO |
|---|---|---|
| Algorithm | DPO (offline) | GRPO (online RL) |
| Critique source | External oracle | Self-generated |
| Training regime | Offline preference pairs | Online rollouts |

These two papers are essentially the same idea arrived at from different training paradigms. Critique-GRPO reports +4.4% on Qwen2.5-7B-Base and +16.7% pass@1 on AIME 2024.

---

## Tier 2 — Same Motivation, Different Implementation

These papers share the core motivation (verifier/critique reasoning improves preference training) but diverge more substantially in mechanism.

### 5. Generative Verifiers / GenRM (Zhang et al., 2024) ★★★★
*Already referenced · arXiv:2408.15240*

The foundational paper for using CoT in verifiers. Establishes that generative next-token-prediction verifiers outperform scalar reward models. Method 1's data source (the oracle's rationale $r$) is exactly what GenRM produces. GenRM generates the $r$; Method 1 uses $r$ as a training signal.

---

### 6. Generative Reward Models / GenRM-STaR-DPO (Mahan et al., 2024) ★★★★
*Already referenced · arXiv:2410.12832*

**Has A + B. Differs on C.**

The STaR-DPO variant applies DPO to (correct rationale + judgement) vs (wrong rationale + judgement) pairs for a judge model. Same verifier-rationale DPO idea as Method 1, but trains a standalone judge.

---

### 7. SuperCorrect (Yang et al., 2024) ★★★★
*NEW · arXiv:2410.09008*

**Has A + C. B is teacher correction trace ≈ verifier rationale.**

Uses cross-model collaborative DPO: a teacher generates error-correction traces; the student base model trains on them via DPO. The teacher's correction trace plays the role of $r$ in Method 1. The difference is that the correction trace is embedded inside the preference pair structure rather than as a separate NLL term.

---

### 8. AIPO (Shen et al., 2024) ★★★
*Already referenced*

**Has A + C. Differs on B.**

DPO + NLL on the winning response $y_w$ directly. The NLL target is a response, not a critique rationale. Structurally close to Method 1 but semantically different — the model memorises good responses rather than internalising reasoning about why they are better.

---

### 9. Self-Taught Evaluators (Wang et al., 2024) ★★★
*Already referenced · arXiv:2408.02666*

Trains evaluators with iterative DPO on reasoning traces. The reasoning trace data (structured as winner/loser evaluations with rationale) is the type of $r$ used in Method 1, but the target model is an evaluator, not the policy model.

---

### 10. CLoud / Critique-out-Loud (Ankner et al., 2024) ★★★
*Already referenced · arXiv:2408.11791*

Trains reward models to generate CoT critiques before scoring. Establishes the value of CoT in the reward signal. Method 1 distils this CoT signal back into the base model rather than keeping it in a separate reward model.

---

### 11. RL Tango (Zha et al., 2025) ★★★
*NEW · arXiv:2505.15034*

Jointly trains generator and generative process verifier with interleaved RL. Verifier CoT emerges as a signal for the generator without process labels. Independently validates Method 1's premise — verifier reasoning is learnable and improves the generator — via a co-training approach rather than a combined loss.

---

## Tier 3 — Important Context (Not Closely Related to the Loss Itself)

| Paper | Why it matters | Status |
|---|---|---|
| Self-Rewarding LMs (Yuan et al., 2024) | Same model acts as generator + judge; motivation for internalising critique | Already referenced |
| Kimi k1.5 (2501.12599) | CoT RM = 98.5% vs 84.4% scalar RM accuracy; empirical case for B | **NEW** |
| DeepSeek-R1 (2501.12948) | CoT reasoning and self-verification emerge from verifiable-reward RL | **NEW** |
| ThinkPRM (2504.16828) | Generative CoT verifier at process level; data-efficient vs scalar PRM | **NEW** |
| Improving RMs w/ Synthetic Critiques (Ye et al., 2024) | Critique data improves reward models; motivates the $r$ data pipeline | Already referenced |

---

## Summary Table

| Paper | A: DPO on answers | B: NLL on verifier CoT | C: Base model | Status |
|---|:---:|:---:|:---:|---|
| **Method 1** | ✓ | ✓ | ✓ | — |
| IRPO (Pang et al., 2024) | ✓ | own CoT, not verifier | ✓ | In paper |
| DJPO (Wang et al., 2024) | ✓ | ✓ | ✗ judge | In paper |
| CFT (Wang et al., 2025) | ✗ | ✓ | ✓ | In paper |
| **Critique-GRPO** (2025) | GRPO | self-critique | ✓ | **New** |
| GenRM-STaR-DPO (Mahan et al., 2024) | ✓ | ✓ | ✗ judge | In paper |
| **SuperCorrect** (2024) | on traces | teacher correction | ✓ | **New** |
| AIPO (2024) | ✓ | NLL on $y_w$ | ✓ | In paper |
| Self-Taught Evaluators (2024) | ✓ | ✓ | ✗ evaluator | In paper |
| CLoud (2024) | ✗ | ✓ | ✗ RM | In paper |
| **RL Tango** (2025) | via RL | verifier CoT | co-training | **New** |

---

## One-Paragraph Positioning (ready to use in the paper)

The closest prior work to Method 1 divides into three groups that each match two of its three defining features. Pang et al. (2024) and AIPO (Shen et al., 2024) use the DPO + NLL loss structure and train the base model, but apply the NLL term to the model's own winning response or reasoning chain rather than an external verifier's critique. Wang et al. (2024b) and Mahan et al. (2024) apply DPO with the verifier's rationale as the NLL target, but train a standalone judge model rather than the base policy. Wang et al. (2025, CFT) fine-tune the base model on critique reasoning via NLL — exactly the second term in our objective — but without a preference term. Concurrently, Critique-GRPO (Zhang et al., 2025) arrives at the same conceptual combination via online GRPO training with self-generated critiques, providing independent empirical validation that joint critique-reasoning and preference optimisation is a productive direction for base model training.
