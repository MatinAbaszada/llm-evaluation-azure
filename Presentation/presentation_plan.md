# Presentation Plan
# "Empirical Evaluation and Cost Optimization of Large Language Models in Azure Cloud Environments"
# Author: Matin Abaszada | Supervisor: Prof. Dr. Ivan Ovsyannikov
# Duration: 15 minutes presentation + 5 minutes Q&A

---

## OVERVIEW SUMMARY

- **Total slides (main deck):** 15
- **Backup slides (for Q&A):** 4
- **Target pace:** ~1 minute per slide on average (some shorter, some longer)
- **Key rule:** Max 3 bullet points per slide. No formulas on main slides (one exception: reward equation shown visually). Let charts speak. Talk more than you read.

---

## COLOR PALETTE

Inspired by Microsoft Azure branding — professional, modern, readable in any room.

| Role                     | Color Name         | Hex Code  |
|--------------------------|--------------------|-----------|
| Slide background         | Pure White         | `#FFFFFF` |
| Header bar (full-width)  | Deep Navy          | `#0B1F4B` |
| Header bar text          | White              | `#FFFFFF` |
| Accent / underlines      | Azure Blue         | `#0078D4` |
| Body text                | Near Black         | `#212121` |
| Highlight / callout box  | Ice Blue           | `#EBF4FF` |
| Positive emphasis        | Microsoft Green    | `#107C10` |
| Negative / failure       | Alert Orange-Red   | `#D83B01` |
| Warning / caution note   | Golden Amber       | `#FFB900` |
| Subtle dividers / lines  | Azure Blue         | `#0078D4` |
| Secondary text / captions| Mid Gray           | `#605E5C` |

### Usage Rules
- All slide titles: Deep Navy bar at top, white text, Azure Blue bottom border
- Section-break slides: Deep Navy background, white text, Azure Blue accent
- Charts: keep their original colors (they are already color-coded and consistent)
- Callout boxes (key insight boxes): Ice Blue background, Azure Blue left border, bold near-black text
- Bullet points: Near Black, no more than 3 per slide
- Progress dots / section indicator at bottom right in Azure Blue

---

## RECOMMENDED FONT

- **Headings:** Segoe UI SemiBold, 28–32pt
- **Body:** Segoe UI Regular, 20–22pt
- **Captions / footnotes:** Segoe UI Italic, 14pt, Mid Gray
- **Callout boxes:** Segoe UI Bold, 20pt

---

## SLIDE-BY-SLIDE PLAN

---

### SLIDE 1 — TITLE SLIDE
**Duration:** ~30 seconds

**Layout:**
- Top 60% of slide: Deep Navy (#0B1F4B) background
- Large white title text
- Bottom 40%: White background with author/supervisor info

**Content:**
- **Title:** Empirical Evaluation and Cost Optimization of Large Language Models in Azure Cloud Environments
- **Author:** Matin Abaszada
- **Supervisor:** Prof. Dr. Ivan Ovsyannikov
- **Program:** Computer Science — Bachelor Thesis
- **Date:** [Defense Date]
- Small Azure Blue horizontal rule between top and bottom sections

**What to say:**
- Briefly introduce yourself and the topic. One sentence: "This thesis evaluates six Azure-hosted LLMs through a unified cost-quality-latency framework and compares three inference optimization strategies."

**No chart on this slide.**

---

### SLIDE 2 — MOTIVATION: WHY DOES THIS MATTER?
**Duration:** ~1 minute 30 seconds

**Layout:**
- 3 visual icon blocks side by side, each with a short label and 1 sentence beneath

**Content:**
Three challenge pillars (use icon + short label):
1. **LLMs are everywhere** — Integrated into coding, Q&A, documentation, enterprise workflows
2. **Azure as the enterprise platform** — Microsoft Foundry: one catalog, pay-per-token, quotas, rate limits
3. **The real problem** — Picking the "strongest" model is not the same as picking the "best" model. Cost, latency, and accuracy all matter simultaneously.

**Callout box (bottom center):**
> "A model that is 93% accurate at $106 per 1000 requests may be a worse deployment choice than one that is 91% accurate at $5 per 1000 requests."

**What to say:**
- Explain that LLMs are deployed in enterprise settings via Azure where you pay per token, have rate limits, and latency is measured. The challenge: academia evaluates models on accuracy alone, but in practice you also pay for speed and cost. That gap is exactly what this thesis addresses.

**No chart on this slide.**

---

### SLIDE 3 — RESEARCH GAP & QUESTIONS
**Duration:** ~1 minute

**Layout:**
- Split: Left half = Gap statement (2 bullets), Right half = 3 numbered RQs

**Content:**

**Gap:**
- Existing literature evaluates models on *single dimensions* (accuracy only, or cost only)
- Optimization strategies (routing, cascading) are tested against *their own internal goal*, not against a unified deployment metric

**Research Questions:**
1. How do Azure-hosted LLMs differ across accuracy, latency, and cost?
2. Is reasoning-enabled inference economically justified?
3. Can a proxy optimization layer (router/cascade/self-consistency) improve cost-efficiency?

**What to say:**
- Keep brief. Say the existing literature doesn't compare all three dimensions together, and that's the gap this thesis fills.

**No chart on this slide.**

---

### SLIDE 4 — STUDY DESIGN (METHODOLOGY OVERVIEW)
**Duration:** ~1 minute 30 seconds

**Layout:**
- Visual flowchart / pipeline diagram (hand-draw a simple one in PowerPoint)
- Three rows: Models → Benchmarks → Strategies

**Content (visual blocks):**

```
  ┌────────────────────────────────────────────────────┐
  │  6 Azure Models (2 generations × 3 tiers)          │
  │  gpt-4.1-mini / gpt-5.4-mini                       │
  │  gpt-4.1 / gpt-5.4                                 │
  │  o3-mini / gpt-5.4-pro (reasoning)                 │
  └──────────────────────────┬─────────────────────────┘
                             │
  ┌──────────────────────────▼─────────────────────────┐
  │  5 Benchmark Datasets                               │
  │  HumanEval · MBPP · MMLU-Pro · GPQA · GSM8K        │
  └──────────────────────────┬─────────────────────────┘
                             │
  ┌──────────────────────────▼─────────────────────────┐
  │  Evaluated Under 4 Strategies                      │
  │  Standalone · Cascade · Router · Self-Consistency  │
  └────────────────────────────────────────────────────┘
```

**Bottom callout:** All strategies evaluated under the **same economic reward function**

**What to say:**
- Everything is compared under the same framework. Six models, five benchmarks, four strategies (one is the baseline). Every result feeds into the same reward function. This is the key design decision that makes comparison fair.

**No chart on this slide (use custom diagram).**

---

### SLIDE 5 — THE ECONOMIC REWARD FRAMEWORK
**Duration:** ~1 minute

**Layout:**
- Center of slide: the reward equation shown as a VISUAL (not LaTeX, use large readable text blocks)
- Below: a short explanation of each component with an icon

**Content:**

```
Reward = − ( Cost  +  λ_latency × Latency  +  λ_error × Error )

          ↑ financial    ↑ response time        ↑ wrong answer
            cost per       (seconds)             (0 or 1)
            request
```

**Three icons below:**
- 💰 **Cost** — deterministic from token counts × Azure prices
- ⏱ **Latency** — wall-clock time per API call
- ✗ **Error** — binary: correct (0) or wrong (1)

**Callout:** "Higher reward = less penalized. Closer to zero is better."

**Note:** λ_latency = 0.01, λ_error = 1.0 by default. Sensitivity analysis was done by sweeping these values.

**What to say:**
- "Instead of comparing models on a single metric, we used this reward function that folds all three deployment concerns into one number. This lets us rank not just individual models, but all optimization strategies on the same scale. The lambda weights let us see how the ranking changes if a business cares more about speed or more about accuracy."

**No chart on this slide (formula displayed as styled text).**

---

### SLIDE 6 — STANDALONE RESULTS: TWO-TIER ACCURACY
**Duration:** ~1 minute 30 seconds

**CHART: chart1_accuracy.png** ← USE THIS ONE

**Layout:**
- Chart takes 70% of the slide
- 2 annotation callouts overlaid on the chart

**Annotations to add on top of chart:**
- Draw a horizontal dashed line at 80% → label: "Tier Boundary: 15 pp gap"
- Left annotation box: "Top Tier: Reasoning-capable (>80%)"
- Right annotation box: "Bottom Tier: Standard models (57–68%)"

**Key numbers to mention:**
- gpt-5.4-pro: 93.2% | gpt-5.4: 90.8% | o3-mini: 83.3%
- gpt-4.1: 68.3% | gpt-5.4-mini: 60.6% | gpt-4.1-mini: 57.6%
- Gap between tiers: ~15 percentage points

**What to say:**
- "The models fall into two clear tiers. The 15 percentage-point gap between the bottom of the top tier and the top of the bottom tier is larger than the spread within either tier. This isn't a gradual improvement — it's a structural jump."

---

### SLIDE 7 — COST vs. ACCURACY: THE PARETO FRONTIER
**Duration:** ~1 minute

**CHART: chart3_cost_vs_accuracy.png** ← USE THIS ONE

**Layout:**
- Chart takes 75% of the slide
- Add one annotation callout

**Annotation to add:**
- Arrow pointing to gpt-5.4: "Best value: 90.8% accuracy at 23× lower cost than gpt-5.4-pro"
- Arrow pointing to gpt-5.4-pro: "770× more expensive than the cheapest model"

**Key message:**
- gpt-5.4 is on the Pareto frontier: high accuracy, moderate cost
- gpt-5.4-pro adds only +2.4pp at ~20× the cost → diminishing returns
- o3-mini and gpt-4.1 are Pareto-dominated

**What to say:**
- "This scatter plot shows cost-accuracy tradeoffs. gpt-5.4 sits at the sweet spot — 90.8% accuracy at moderate cost. gpt-5.4-pro is far to the right: it adds only 2.4 percentage points over gpt-5.4 while costing twenty times more. The cost difference between the cheapest and most expensive model spans almost three orders of magnitude."

---

### SLIDE 8 — ECONOMIC REWARD: THE REAL RANKING
**Duration:** ~1 minute

**CHART: chart4_reward.png** ← USE THIS ONE

**Layout:**
- Chart takes 70% of slide
- Callout box on the side

**Callout box:**
> "Most accurate ≠ Best reward. gpt-5.4-pro (93.2% acc) ranks 5th in reward. gpt-5.4 (90.8% acc) ranks 1st."

**Key contrast to highlight:**
- gpt-5.4-pro: #1 accuracy → #5 reward (penalized by cost + latency)
- o3-mini: #3 accuracy → #6 reward (penalized by 65.7s mean latency)
- gpt-5.4: #2 accuracy → #1 reward

**What to say:**
- "When we fold cost and latency into the reward function, the ranking changes significantly. The most accurate model — gpt-5.4-pro — drops to fifth place because its additional 2.4 percentage points of accuracy doesn't compensate for the extra $100 per 1000 requests and 20 extra seconds of latency. gpt-5.4 is the most balanced choice."

---

### SLIDE 9 — WHERE IS EACH MODEL THE BEST? (DECISION MAP)
**Duration:** ~1 minute

**CHART: chart7_best_model_heatmap.png** ← USE THIS ONE

**Layout:**
- Chart takes the full center (80% of slide)
- Small legend explanation at bottom

**Explanation caption below chart:**
> "Each cell = which model wins at those (λ_error, λ_latency) weights. The star ★ marks our default setting."

**Key regions to point out verbally:**
- Top-left (high latency penalty): gpt-4.1-mini wins (fastest)
- Center (default zone): gpt-5.4 wins (best overall balance)
- Bottom-right (high error penalty): gpt-5.4-pro wins (most accurate)

**What to say:**
- "This heatmap answers: 'which model should I choose?' The answer depends on your deployment priorities. If your system is latency-sensitive, use a mini model. If cost and accuracy both matter at reasonable speed, gpt-5.4 is the winner across the broadest region. If you need maximum accuracy and latency/cost is not a concern, gpt-5.4-pro. This shows that model selection must be context-dependent."

---

### SLIDE 10 — OPTIMIZATION STRATEGY 1: CASCADE (FAILED BY OVERCONFIDENCE)
**Duration:** ~1 minute 15 seconds

**CHART: chartC2_cascade_dual_heatmap.png** ← USE THIS ONE

**Layout:**
- Left: Brief explanation of cascade strategy (diagram: Small Model → [confidence gate] → Large Model)
- Right: The dual heatmap (reward + cost)

**Custom mini-diagram (left side):**
```
Query → [Small Model]
           ↓
     Confidence ≥ T?
     YES → Return answer
     NO  → [Large Model] → Return answer
```

**Key finding callout:**
> ⚠ Maximum escalation rate was only 7.3% even at T=90. Modern LLMs are overconfident — the gate almost never opens.

**What to say:**
- "The cascade strategy sounds logical: let a cheap model answer easy questions, escalate hard ones to a stronger model. The problem is that current Azure-hosted LLMs almost never report low confidence. Even at the most aggressive threshold, only 7.3% of queries escalated. So every cascade configuration behaves almost identically to its small model alone — no accuracy gain, just marginally higher latency."

---

### SLIDE 11 — OPTIMIZATION STRATEGY 2: ROUTER (THE WINNER)
**Duration:** ~1 minute 30 seconds

**CHART: chartR3_router_best_config_heatmap.png** ← USE THIS ONE

**Layout:**
- Left: Brief explanation of router (diagram: Router classifies → Small or Large)
- Right: The decision heatmap (router wins a substantial band)

**Custom mini-diagram (left side):**
```
Query → [Router Model (gpt-5.4-mini)]
           ↓
     EASY? → [Small: gpt-4.1-mini]
     HARD? → [Large: gpt-4.1]
```

**Key achievement callout:**
> ✔ Best router: +15.5 pp accuracy over small baseline, reward of −0.498 vs. −0.527 for gpt-4.1 standalone.
> The router wins a large diagonal region of the decision map — the ONLY optimization to do so.

**Numbers to mention:**
- Router accuracy: 73.1% (vs small model 57.6%)
- Router cost: 0.344 m$ (vs large model 0.687 m$)
- Router reward: −0.498 (beats all standalones except gpt-5.4)

**What to say:**
- "The router uses an external classifier to decide which model answers each query. Unlike the cascade, the router doesn't rely on the model's own confidence. The best configuration — using gpt-5.4-mini as router, routing between gpt-4.1-mini and gpt-4.1 — achieved 73% accuracy while spending only half the cost of the large model standalone. It's the only optimization strategy in the study that wins a substantial region of the decision map, beating all six standalone models in the middle penalty range."

---

### SLIDE 12 — OPTIMIZATION STRATEGY 3: SELF-CONSISTENCY (MODEST GAINS)
**Duration:** ~1 minute

**CHART: chartSC5_selfcons_accuracy_comparison.png** ← USE THIS ONE

**Layout:**
- Chart takes 70% of slide
- Callout box with key trade-off

**Custom mini-diagram (top left, small):**
```
Query → [Model] × 3 (parallel)
         ↓
    Majority Vote → Final Answer
```

**Callout box:**
> ✔ gpt-4.1-mini N=3: +6.8 pp accuracy (27.8% → 44.0% on MMLU-Pro!)
> ⚠ Cost: exactly 3× more expensive. Doesn't reach the next standalone tier (68.3%).

**What to say:**
- "Self-consistency samples the same model three times in parallel and takes the majority answer. It meaningfully improves accuracy — especially on hard multiple-choice benchmarks like MMLU-Pro, from 28% to 44%. The cost is exactly 3× and latency is only 1.07× (because calls are parallel). However, even with this improvement, it never closes the gap to the next standalone model tier."

---

### SLIDE 13 — CROSS-STRATEGY COMPARISON TABLE
**Duration:** ~1 minute

**Layout:**
- Clean summary table, centered on slide
- Each row highlighted by best performance per column

**Table:**

| Strategy          | Best Configuration                   | Accuracy | Latency | Cost (m$) | Reward   |
|-------------------|--------------------------------------|----------|---------|-----------|----------|
| **Standalone**    | gpt-5.4                              | 90.8%    | 34.7s   | 5.370     | **−0.444** |
| **Router**        | rtr=gpt-5.4-mini → gpt-4.1-mini/4.1 | 73.1%    | 22.9s   | 0.344     | −0.498   |
| **Self-Consist.** | gpt-4.1-mini N=3                     | 64.4%    | 18.6s   | 0.765     | −0.543   |
| **Cascade**       | gpt-4.1-mini → gpt-5.4 T=75         | 57.7%    | 17.7s   | 0.261     | −0.601   |

Color coding for cells:
- Reward column: green = best, orange = worst
- Cost: green = cheapest, red = most expensive
- Accuracy: green = highest

**Callout:**
> "Only the Router beats all six standalone models on a substantial region of the decision map."

**What to say:**
- "Here's the head-to-head summary. gpt-5.4 standalone has the best reward, but the router comes close — at 16× lower cost and 12 seconds faster. Cascade is the weakest — barely different from its small model. This table shows that optimization can work, but only when the escalation logic is well-calibrated."

---

### SLIDE 14 — CONCLUSIONS & ANSWERS TO RESEARCH QUESTIONS
**Duration:** ~1 minute 30 seconds

**Layout:**
- Three numbered sections, each with RQ label + one-line answer + brief detail

**Content:**

**RQ1: How do Azure LLMs differ in accuracy, latency, and cost?**
→ Two-tier structure: 15pp accuracy gap, 770× cost gap. No model Pareto-dominates. **gpt-5.4** is the best balanced choice at default weights.

**RQ2: Is reasoning-enabled inference economically justified?**
→ Only for accuracy-critical, latency-tolerant tasks. gpt-5.4-pro adds +2.4pp at 20× the cost and 1.5× the latency — **not justified** for general workloads.

**RQ3: Can proxy optimization improve cost-efficiency?**
→ **Router: YES** — wins a diagonal band in the decision map, beating all standalones at middle penalty weights.
→ **Self-Consistency: partially** — modest accuracy lift at 3× cost, parallel execution preserves latency.
→ **Cascade: NO** — neutralized by LLM overconfidence. Almost never escalates.

**What to say:**
- Walk through each answer concisely. Emphasize that the answer to each RQ is context-dependent — the decision map captures this explicitly.

**No chart on this slide.**

---

### SLIDE 15 — LIMITATIONS, FUTURE WORK & THANK YOU
**Duration:** ~1 minute

**Layout:**
- Two columns: Limitations (left) | Future Work (right)
- Bottom: "Thank you" + contact

**Limitations:**
- 6 models from gpt-4.1/5.4/o3 families only; other Azure models not covered
- 5 benchmark families; no long-context or multi-turn dialogue
- Latency measured under one quota/region — may vary
- Cascade used raw confidence without post-hoc calibration

**Future Work:**
- Calibrated cascade (post-hoc or verifier-based escalation)
- Lightweight dedicated router models; hybrid strategies
- Extended benchmarks: domain-specific, long-context
- Online deployment framework with real-time quota signals

**Bottom of slide:**
> Thank you! Questions welcome.

**What to say:**
- Brief mention of limitations, then invite questions. Be ready with backup slides.

---

## BACKUP SLIDES (for Q&A)

---

### BACKUP SLIDE B1 — Per-Dataset Accuracy Breakdown
**CHART: chart2_accuracy_per_dataset.png**
- Use this if asked "which model is best for coding tasks?" or "how did o3-mini do on specific datasets?"
- Key insight: coding benchmarks are saturated (all models 80–100%). GPQA and MMLU-Pro discriminate best.

---

### BACKUP SLIDE B2 — Self-Consistency Decision Map
**CHART: chartSC3_selfcons_best_config_heatmap.png**
- Use this if asked about when self-consistency is worth it
- gpt-4.1-mini N=3 dominates the upper-left (latency-tolerant, accuracy-important) region

---

### BACKUP SLIDE B3 — Router Detail: Routing Behavior by Dataset
**CHART: chartR1_router_overview.png**
- Use this if asked how the router decides which questions to escalate
- Shows that coding tasks are almost never escalated; GPQA gets 40%+ escalation rate

---

### BACKUP SLIDE B4 — Reward Formula Full Derivation
- Show the full reward equation with all parameters
- Explain why cost is in dollars (not milli-dollars) and how it enters the reward space
- Use if asked about the technical detail of the economic framework

---

## CHART SELECTION GUIDE

### APPROVED FOR MAIN DECK (clean, readable at a glance)

| Chart File                        | Used On Slide | Why Chosen |
|-----------------------------------|---------------|------------|
| chart1_accuracy.png               | Slide 6       | Simple bar chart, two tiers immediately visible |
| chart3_cost_vs_accuracy.png       | Slide 7       | Scatter plot, Pareto frontier instantly clear |
| chart4_reward.png                 | Slide 8       | Bar chart, clean ranking |
| chart7_best_model_heatmap.png     | Slide 9       | 2D decision map, message is a clear spatial region |
| chartC2_cascade_dual_heatmap.png  | Slide 10      | Compact heatmap, reward + cost at once |
| chartR3_router_best_config_heatmap.png | Slide 11 | Decision map showing router's winning region |
| chartSC5_selfcons_accuracy_comparison.png | Slide 12 | Grouped bar chart, before/after SC clearly visible |

### APPROVED FOR BACKUP SLIDES ONLY

| Chart File                           | Backup Slide | Why Not Main Deck |
|--------------------------------------|--------------|-------------------|
| chart2_accuracy_per_dataset.png      | B1           | 6 bars × 5 groups — too dense for main deck |
| chartSC3_selfcons_best_config_heatmap.png | B2      | Needs context from SC explanation |
| chartR1_router_overview.png          | B3           | 4 subplots — too complex for one main slide |

### NOT RECOMMENDED FOR PRESENTATION

| Chart File                                | Reason |
|-------------------------------------------|--------|
| chart5_reward_vs_lambda_error.png         | Sensitivity sweep — too technical for general audience |
| chart6_reward_vs_lambda_latency.png       | Same reason |
| chart8_best_model_heatmap_per_dataset.png | 5 heatmaps side by side — unreadable in presentation room |
| chartC1_cascade_combined_overview.png     | 4 dual-axis subplots — too complex |
| chartC3_cascade_best_config_heatmap.png   | Cascade wins no cells — confusing without context |
| chartC4_cascade_reward_vs_lambda_error.png | Sensitivity sweep — too technical |
| chartC5_cascade_reward_vs_lambda_latency.png | Same reason |
| chartC6_cascade_per_dataset_decision_map.png | Too complex |
| chartR2_router_dual_heatmap.png           | Replaced by R3 which tells a stronger story |
| chartR4_router_reward_vs_lambda_error.png | Sensitivity sweep |
| chartR5_router_reward_vs_lambda_latency.png | Same reason |
| chartR6_router_per_dataset_decision_map.png | Too complex |
| chartSC1_selfcons_overview.png            | Replaced by SC5 which is cleaner |
| chartSC2_selfcons_dual_heatmap.png        | Less impactful than SC5 |
| chartSC4_selfcons_reward_vs_lambda_error.png | Sensitivity sweep |

---

## TIMING BREAKDOWN

| Slide | Topic                              | Time      |
|-------|------------------------------------|-----------|
| 1     | Title                              | 0:00–0:30 |
| 2     | Motivation & Context               | 0:30–2:00 |
| 3     | Research Gap & RQs                 | 2:00–3:00 |
| 4     | Study Design Overview              | 3:00–4:30 |
| 5     | Economic Reward Framework          | 4:30–5:30 |
| 6     | Standalone Accuracy (chart1)       | 5:30–7:00 |
| 7     | Cost vs. Accuracy (chart3)         | 7:00–8:00 |
| 8     | Economic Reward Rankings (chart4)  | 8:00–9:00 |
| 9     | Decision Map (chart7)              | 9:00–10:00|
| 10    | Cascade (chartC2)                  | 10:00–11:15|
| 11    | Router (chartR3)                   | 11:15–12:45|
| 12    | Self-Consistency (chartSC5)        | 12:45–13:45|
| 13    | Cross-Strategy Summary Table       | 13:45–14:45|
| 14    | Conclusions & RQ Answers           | 14:45–15:00 (tight — can trim slide 12 by 15s)|
| 15    | Limitations & Thank You            | last ~30s  |

**TOTAL: 15 minutes**

---

## DESIGN PRINCIPLES TO FOLLOW

1. **One key message per slide.** State it in the slide title, show it with the chart, say it aloud.
2. **Max 3 bullet points per slide.** If you have more, move them to a backup slide.
3. **Charts fill ≥ 65% of the slide.** Don't shrink them — let the visual do the work.
4. **Add annotation callouts** (arrows, text boxes in Azure Blue or Amber) directly on charts to guide the audience's eye.
5. **No raw formulas on main slides.** The reward equation is shown as a styled typographic layout, not LaTeX output.
6. **Section-break slides** (before Methodology, Results, Conclusion) use Deep Navy full-background with white text. These add structure without eating time.
7. **Progress indicator** (small dots or section label at bottom right) helps audience track where you are.
8. **Consistent icon style** — use Fluent/Microsoft-style icons or simple flat icons in Azure Blue.

---

## SLIDE TITLE SUGGESTION LIST

| Slide | Suggested Title |
|-------|-----------------|
| 1     | (just thesis title + author) |
| 2     | Why Choosing the Right Azure LLM Is Not Trivial |
| 3     | Research Gap & Questions |
| 4     | Study Design at a Glance |
| 5     | The Unified Evaluation Framework |
| 6     | Accuracy: A Two-Tier Structure |
| 7     | Cost vs. Accuracy: 770× Difference |
| 8     | Economic Reward: The Ranking Changes |
| 9     | Which Model Wins — and When? |
| 10    | Cascade: Blocked by Overconfidence |
| 11    | Router: The Only Strategy That Wins |
| 12    | Self-Consistency: Modest but Predictable Gains |
| 13    | Head-to-Head: Best Configuration per Strategy |
| 14    | Conclusions: Answering the Research Questions |
| 15    | Limitations & Future Work |

---
*Generated: May 2026 | For the defense of: "Empirical Evaluation and Cost Optimization of LLMs in Azure Cloud Environments"*
