<h2 align="center">
OptiSkill: A Hierarchical and Evolving SkillBank for LLM-Based Optimization Modeling
</h2>

<p align="center">
  <b>Skill-augmented operations research modeling with reusable, solver-verified formulation knowledge</b>
</p>

---

This repository contains the official code, data, prompts, and supplementary materials for the paper:

> **OptiSkill: A Hierarchical and Evolving SkillBank for LLM-Based Optimization Modeling**

We propose **OptiSkill**, a skill-augmented framework for automated operations research (OR) modeling. Instead of solving each optimization problem in isolation, OptiSkill accumulates solver-verified modeling experience into a hierarchical **SkillBank** and retrieves reusable skills to guide future formulations.

OptiSkill represents OR modeling knowledge as two complementary skill types:

- **Global Strategies** capture problem-level formulation skeletons, such as flow balance, time-indexed planning, routing structures, and allocation procedures.
- **Step Experiences** capture local error-prevention rules, such as variable-domain checks, Big-M linking, boundary conditions, and solver-consistent constraint design.

The SkillBank is further improved through **stable batch-level test-time evolution**, where newly discovered and repaired skills are buffered, validated, and incorporated only when they pass support, repair, and regression checks.

---

## 🧠 Framework Overview

<div align="center">
  <img src="assets/framework.pdf" width="900"/>
  <p><em>
  Figure 1: Overview of OptiSkill. OptiSkill constructs a hierarchical SkillBank from verified modeling trajectories, retrieves relevant skills for OR formulation, and evolves the SkillBank at test time.
  </em></p>
</div>

OptiSkill consists of three main modules:

**1. OR Modeling SkillBank Construction**
- Collect solver-verified modeling trajectories
- Distill **Global Strategies** from correct trajectories
- Distill **Step Experiences** from contrastive correct-error trajectory pairs
- Deduplicate and merge candidates within each OR problem-type bucket

**2. Skill-Augmented OR Formulation**
- Identify the dominant OR problem type
- Retrieve relevant strategies and experiences from the corresponding sub-bank
- Generate solver-ready mathematical formulations and Gurobi code with compact skill guidance

**3. Stable Test-Time SkillBank Evolution**
- Diagnose failed formulations with solver feedback and self-refinement
- Generate new skills for missing knowledge
- Repair misleading skills when retrieved guidance causes failures
- Validate updates at the batch level before modifying the active SkillBank

---

## 🧩 SkillBank Design

OptiSkill organizes reusable OR modeling knowledge by problem type:

```text
SkillBank
├── allocation
├── assignment
├── selection
├── flow
├── time_planning
├── routing
├── scheduling
└── special
```

Each sub-bank contains two kinds of skills.

### Global Strategy

A Global Strategy describes when a formulation pattern applies and how to construct the model at a high level.

```json
{
  "summary": "when this strategy is applicable",
  "procedure": [
    "define core entities and indices",
    "introduce decision variables",
    "construct the objective",
    "add structural constraints",
    "check domains and feasibility logic"
  ]
}
```

### Step Experience

A Step Experience captures a local high-risk modeling situation and the corresponding corrective action.

```json
{
  "trigger": "local structural signal or potential modeling risk",
  "guidance": "warning and corrective modeling action"
}
```

This design makes SkillBank knowledge compact, interpretable, composable, and directly usable during formulation generation.

---

## 🔥 Main Results: Pass@1 Accuracy

**Table 1. Overall Pass@1 accuracy (%) on eight optimization modeling benchmarks.**  
Best results among agentic methods are shown in **bold**.

| Backbone | Method | NL4OPT | MAMO Easy | MAMO Cpx. | NLP4LP | OptiB. | CpxOR | IndOR | OptM. | Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DouBao-Seed-2.0 | CoE | 78.4 | 91.6 | 63.1 | 94.3 | 84.4 | 50.0 | 64.3 | 34.9 | 70.1 |
| DouBao-Seed-2.0 | OptiMUS | 75.1 | 88.4 | 45.9 | 89.9 | 84.6 | 44.4 | 61.9 | 31.9 | 65.3 |
| DouBao-Seed-2.0 | OptiTree | 76.5 | 92.1 | 62.2 | 91.0 | 83.1 | 44.4 | 69.0 | 38.6 | 69.6 |
| DouBao-Seed-2.0 | AlphaOPT | 75.6 | 91.4 | 55.0 | 93.3 | 85.1 | 55.6 | 66.7 | 36.1 | 69.9 |
| DouBao-Seed-2.0 | ReLoop | 76.1 | 91.2 | 62.2 | 92.7 | 84.9 | 50.0 | 59.5 | 32.5 | 68.6 |
| DouBao-Seed-2.0 | **OptiSkill** | **78.9** | 90.8 | **64.0** | 93.8 | **85.4** | **61.1** | **71.4** | **40.4** | **73.2** |
| DeepSeek-V4 | CoE | 78.9 | **96.1** | 57.7 | 92.1 | 82.6 | 50.0 | 66.7 | 39.2 | 70.4 |
| DeepSeek-V4 | OptiMUS | 77.5 | 88.4 | 38.7 | 88.2 | 83.1 | 44.4 | 59.5 | 31.3 | 63.9 |
| DeepSeek-V4 | OptiTree | 80.3 | 93.9 | **60.4** | 93.8 | 86.9 | 50.0 | 71.4 | 41.0 | 72.2 |
| DeepSeek-V4 | AlphaOPT | 76.1 | 95.8 | 54.1 | 90.4 | 80.6 | 55.6 | 64.3 | 39.2 | 69.5 |
| DeepSeek-V4 | ReLoop | 74.7 | 91.6 | 56.8 | 89.3 | 79.9 | 55.6 | 66.7 | 27.1 | 67.7 |
| DeepSeek-V4 | **OptiSkill** | **80.8** | 94.3 | 56.8 | **94.9** | **87.6** | **61.1** | **73.8** | **41.6** | **73.9** |

**Key observations**:

- OptiSkill achieves the strongest macro average among agentic methods under both backbones.
- Gains are more pronounced on structurally challenging benchmarks such as ComplexOR, IndustryOR, and OptMATH.
- Explicit strategies and local experiences are especially useful for long, fragile formulation chains.

---

## 📈 Test-Time SkillBank Evolution

<div align="center">
  <img src="assets/res_evolve.pdf" width="850"/>
  <p><em>
  Figure 2: Effect of static and evolved SkillBank across benchmarks.
  </em></p>
</div>

OptiSkill further improves its SkillBank during test-time use. Compared with the static SkillBank, test-time evolution brings additional gains on several benchmarks, including:

| Benchmark | Gain over Static SkillBank |
|---|---:|
| NL4OPT | +4.2 |
| OptiBench | +3.7 |
| OptMATH | +7.2 |

During evolution, OptiSkill diagnoses failed attempts and separates them into several categories:

- `strategy_miss`: missing global formulation skeleton
- `experience_miss`: missing local anti-error guidance
- `strategy_misleading`: retrieved strategy is too broad or structurally misleading
- `experience_misleading`: retrieved experience has an overly broad trigger or misleading guidance
- `code_error_only`: failure mainly comes from implementation rather than modeling
- `unresolved`: failure source is not reliable enough for skill update

Only validated new or repaired skills are added to the active SkillBank.

---

## 📁 Repository Structure

```text
OptiSkill/
├── assets/                     # Figures used in README and paper
├── data/                       # Benchmark data and verified trajectories
├── skillbank/                  # Initial and evolved SkillBank files
│   ├── init/
│   └── evolved/
├── template/                   # Distillation, retrieval, formulation, and evolution prompts
├── scripts/                    # Build, evaluate, and evolve scripts
└── README.md
```

---

## ✅ Why OptiSkill?

OptiSkill is designed for reliable and adaptive OR modeling with frozen LLMs.

- **Reusable**: converts historical solver-verified trajectories into reusable modeling skills
- **Hierarchical**: separates problem-level strategies from step-level experiences
- **Composable**: retrieves compact skills that can be combined across formulation steps
- **Evolvable**: updates the SkillBank through validated batch-level test-time evolution
- **Interpretable**: keeps explicit skill fields that are inspectable and editable
- **Efficient**: avoids heavy multi-agent exploration by using compact formulation guidance
