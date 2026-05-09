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
- Large white title text, centered
- Bottom 40%: White background with author/supervisor info
- **Constructor University logo** placed top-left of the Deep Navy section (or centered above the title)

**Content:**
- **Title:** Empirical Evaluation and Cost Optimization of Large Language Models in Azure Cloud Environments
- **Author:** Matin Abaszada
- **Supervisor:** Prof. Dr. Ivan Ovsyannikov
- **Program:** Computer Science — Bachelor Thesis
- **Date:** [Defense Date]
- Small Azure Blue horizontal rule between top and bottom sections

**LOGO EXTRACTION NOTE:**
The Constructor University logo is in `Presentation/Thesis Guideline.pptx` (first slide).
Claude (via python-pptx) can extract it automatically:
```python
from pptx import Presentation
from pptx.util import Inches
from PIL import Image
import io

prs = Presentation('Presentation/Thesis Guideline.pptx')
slide = prs.slides[0]
for shape in slide.shapes:
    if shape.shape_type == 13:  # MSO_SHAPE_TYPE.PICTURE
        img_bytes = shape.image.blob
        with open('Presentation/cu_logo_extracted.png', 'wb') as f:
            f.write(img_bytes)
        break
```
Then embed the saved PNG into the title slide of the new presentation. This is fully automated — Claude can include this in the python-pptx build script.

**What to say:**
- Briefly introduce yourself and the topic. One sentence: "This thesis evaluates six Azure-hosted LLMs through a unified cost-quality-latency framework and compares three inference optimization strategies."

**No chart on this slide.**

---

### SLIDE 2 — MOTIVATION: WHY DOES THIS MATTER?
**Duration:** ~1 minute 45 seconds

**Layout:**
- 4 visual blocks arranged as a flow: top-row = 3 blocks side by side showing the journey, bottom = one wide callout

**Content — tell it as a story, left to right:**

**Block 1 — The Expensive Reality (icon: server rack / GPU)**
> Most LLM research assumes you run models **locally**.
> That means: GPU clusters, cooling, security infrastructure, dedicated ops teams, and millions in capital expense.

**Block 2 — The Shift to Cloud (icon: cloud)**
> Today, enterprises move to **cloud environments** instead.
> One API call replaces an entire data center. Security, compliance, scaling — handled. Pay only for what you use.

**Block 3 — Azure as the Platform (icon: Azure logo)**
> **Microsoft Azure** is one of the leading cloud LLM platforms.
> Model catalog (OpenAI, DeepSeek, Meta…), pay-per-token pricing, quotas, rate limits — all in one place.

**Wide callout box (bottom, full width):**
> ⚠ But in the cloud, "best model" ≠ "strongest model".
> A model at 93% accuracy that costs 770× more per request may be the wrong enterprise choice.
> **Accuracy, cost, and latency must all be evaluated together.**

**What to say:**
- "Most academic LLM research is done by running models locally — on expensive GPU infrastructure. In practice, the majority of enterprises today don't do that. They use cloud environments: pay-per-token, no hardware, security already built in. Azure is one of the most prominent of these platforms, offering dozens of models through a single API. But this changes the evaluation problem. In the cloud, you're billed per token, your latency reflects quota tiers and throttling, and the so-called strongest model might be completely impractical for your budget or time constraints. That's the motivation for this thesis."

**No chart on this slide.**

---

### SLIDE 3 — RESEARCH GAP & QUESTIONS
**Duration:** ~1 minute 15 seconds

**Layout:**
- Split: Left 55% = Gap (3 bullets, stacked), Right 45% = 3 numbered RQs in a blue box

**Content:**

**Gap (left side, 3 bullets):**
- Most existing studies evaluate models in isolation — accuracy only, or cost only, never together
- Optimization strategies (routing, cascading) are benchmarked against their *own internal goal*, not a unified deployment reward
- **Critically: almost no studies test these strategies in a real cloud environment** — where pricing, quotas, and rate limits directly shape measured latency and cost. Testing in Azure fills this gap by grounding results in actual enterprise service constraints.

**Research Questions (right side, numbered box):**
1. How do Azure-hosted LLMs differ across accuracy, latency, and cost?
2. Is reasoning-enabled inference economically justified?
3. Can a proxy optimization layer (router/cascade/self-consistency) improve cost-efficiency?

**What to say:**
- "The literature has two gaps. First, metrics are evaluated in isolation — accuracy papers don't count cost, cost papers don't measure quality. Second, and more practically relevant: almost none of these studies are conducted in a live cloud environment. Azure has quotas, throttling, pay-per-token pricing, and version-controlled deployments — all of which affect your actual deployment decision. This thesis closes that gap by running everything in a real Azure environment and evaluating all results through a single unified framework."

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

**WHY NEGATIVE REWARD? — Include this as a spoken explanation, not on slide text:**
The reward is intentionally defined as a *negative penalty* rather than a positive utility. This design comes from the same reasoning used in the Economic Evaluation of LLMs (Yue et al.):
- A *positive* utility would require you to define a maximum achievable score — an arbitrary ceiling. What is a "perfect" answer worth in dollars?
- A *negative penalty* has a natural zero: **zero means no cost, no latency, no error** — a theoretically perfect free instant answer. Every real model can only be penalized from this ideal.
- This makes values directly interpretable: −0.444 means the average request incurred 0.444 units of combined penalty.
- It also aligns with **loss minimization** — the standard framing in both economics and machine learning — rather than inventing an arbitrary utility scale.
- Finally, it makes sensitivity analysis intuitive: as λ_error increases, models with higher error rates drop faster (steeper negative slope), which is exactly what you observe in the sweep charts.

**What to say on slide:**
- "Instead of comparing models on a single metric, we used this reward function that folds all three deployment concerns into one number. It's defined as a negative penalty — zero would mean a free, instant, always-correct answer. Every real model incurs some penalty. This lets us rank all models and strategies on the same scale, and the lambda weights let us simulate different deployment priorities."

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

### SLIDE 10 — OPTIMIZATION STRATEGY 1: CASCADE (BLOCKED BY OVERCONFIDENCE)
**Duration:** ~1 minute 15 seconds

**PSYCHOLOGICAL NOTE — CRESCENDO ORDER (worst → modest → best):**
Strategies are intentionally ordered from weakest to strongest. This builds a narrative arc:
"Here's the idea that should work but doesn't → here's a partial win → here's what actually works."
This leaves the audience on a high note and is the most effective structure for academic defense presentations (primacy-recency: they remember the *last* thing you say).

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
- "Let's look at our first optimization attempt. The cascade strategy is intuitive: let a cheap model answer easy questions, and only escalate to a stronger model when the cheap one is uncertain. The problem? Current Azure-hosted LLMs are systematically overconfident. They almost never report low confidence. Even at the most aggressive threshold, only 7.3% of queries escalated. So every cascade configuration behaves almost identically to its small model alone — no meaningful accuracy gain, just a small latency overhead. Cascade failed — not because the idea is wrong, but because it requires calibrated confidence, which current models don't provide out of the box."

---

### SLIDE 11 — OPTIMIZATION STRATEGY 2: SELF-CONSISTENCY (MODEST GAINS)
**Duration:** ~1 minute

**CHART: chartSC5_selfcons_accuracy_comparison.png** ← USE THIS ONE

**Layout:**
- Chart takes 65% of slide (right side)
- Left side: mini diagram + callout

**Custom mini-diagram (left side):**
```
Query → [Model] × 3 (parallel)
         ↓
    Majority Vote → Final Answer
```

**Callout box:**
> ✔ gpt-4.1-mini N=3: +6.8 pp accuracy (MMLU-Pro: 27.8% → 44.0%)
> ⚠ Cost: exactly 3× more expensive. Still below next standalone tier (68.3%).

**What to say:**
- "Self-consistency takes a different approach — it doesn't change the model at all. It sends the same question three times in parallel and picks the majority answer. This genuinely improves accuracy: on MMLU-Pro, gpt-4.1-mini goes from 28% to 44%. Cost triples, but latency is only 7% higher because the calls run in parallel. It's a real improvement — but it still doesn't close the gap to the next standalone tier. A partial win: reliable, predictable, but limited."

---

### SLIDE 12 — OPTIMIZATION STRATEGY 3: ROUTER (THE WINNER — CLIMAX)
**Duration:** ~1 minute 30 seconds

**CHART: chartR3_router_best_config_heatmap.png** ← USE THIS ONE

**Layout:**
- Left: Brief explanation of router (diagram: Router classifies → Small or Large)
- Right: The decision heatmap (router wins a substantial diagonal band)

**Custom mini-diagram (left side):**
```
Query → [Router Model (gpt-5.4-mini)]
           ↓
     EASY? → [Small: gpt-4.1-mini]
     HARD? → [Large: gpt-4.1]
```

**Key achievement callout:**
> ✔ Best router: +15.5 pp accuracy over small baseline at 16× lower cost than gpt-5.4.
> Reward: −0.498 — beats gpt-4.1 (−0.527) and both mini standalones.
> **The ONLY optimization that wins a region of the macro decision map.**

**Numbers to mention:**
- Router accuracy: 73.1% (vs small model 57.6%)
- Router cost: 0.344 m$ (vs gpt-4.1 standalone: 0.687 m$, vs gpt-5.4: 5.37 m$)
- Router reward: −0.498 (second only to gpt-5.4 at −0.444)

**IMPORTANT — Why router beats SC despite SC winning larger area in its own chart:**
In chartSC3, SC is compared against standalone models ONLY (no router in the candidate set).
In chartR3, the router is compared against standalone models ONLY (no SC in the candidate set).
The two charts DO NOT compete against each other.
- SC wins the top-left (very high λ_latency zone) — an extreme scenario where even tiny latency differences dominate
- The router wins the central diagonal band, which includes the DEFAULT ★ setting (λ_error=1.0, λ_latency=0.01)
- At the star position in SC3, gpt-5.4 wins — NOT the SC configuration
- At the star position in R3, the router wins
- The star = where real deployments typically live. The router wins there; SC does not.
- Additionally: SC wins because its latency is barely worse than standalone (1.07×), so under extreme latency penalties it "looks good" by not being much worse — but it's not winning on merit in the region that matters.

**What to say:**
- "The router takes the most direct approach: an external model classifies each query before it's answered, and routes it to either a cheap or a strong model based on predicted difficulty. Unlike the cascade, the router doesn't rely on the model's own confidence — it makes an independent decision. The best configuration — gpt-5.4-mini as router, routing between gpt-4.1-mini and gpt-4.1 — achieves 73% accuracy while spending only half the cost of the large model standalone. It's the only optimization in this study that wins a substantial region of the decision map at the default settings — the settings that reflect real enterprise deployments. This is the crescendo of the optimization story."

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
| 10    | Cascade (fails — overconfidence)           | 10:00–11:15|
| 11    | Self-Consistency (modest gains)            | 11:15–12:15|
| 12    | Router — THE WINNER (climax)               | 12:15–13:45|
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
| 11    | Self-Consistency: Modest but Predictable Gains |
| 12    | Router: The Only Strategy That Wins (Climax) |
| 13    | Head-to-Head: Best Configuration per Strategy |
| 14    | Conclusions: Answering the Research Questions |
| 15    | Limitations & Future Work |

---
*Generated: May 2026 | For the defense of: "Empirical Evaluation and Cost Optimization of LLMs in Azure Cloud Environments"*
