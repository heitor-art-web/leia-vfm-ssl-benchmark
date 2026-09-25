# Literature map — semi-supervised medical segmentation in the VFM era

Search snapshot: **2026-09-25**.

This is a curated map, not a claim of an exhaustive census. The repository is designed as a **living review**: every paper should be recorded in `literature/literature_matrix.csv` with method family, dataset, annotation regime, foundation model, code availability and relevance to the benchmark.

## 1. Classical semi-supervised learning primitives

The current VFM-era methods still reuse a small set of ideas:

- pseudo-labeling;
- consistency regularization;
- teacher–student / EMA teachers;
- uncertainty filtering;
- cross pseudo supervision;
- adaptive confidence thresholds;
- co-training / heterogeneous learners.

Canonical anchors include Mean Teacher (2017), UA-MT (MICCAI 2019), FixMatch (NeurIPS 2020), SASSNet (MICCAI 2020), Deep Co-Training (Pattern Recognition 2020), DTC (AAAI 2021), CPS (CVPR 2021) and FreeMatch (ICLR 2023).

## 2. Medical SSL benchmark infrastructure

**SSL4MIS** is a major open benchmark/codebase and literature index. It includes implementations or references for Mean Teacher, entropy minimisation, adversarial SSL, UA-MT, ICT, URPC, CPS, CCT, co-training, CNN/Transformer cross teaching, FixMatch and others.

Implication for this repository: **do not re-implement ten classical methods at first**. Start with one transparent SSL primitive (Mean Teacher), then add methods only if the result needs triangulation.

## 3. Vision foundation models for medical segmentation

### MedSAM (2024)

Medical adaptation of SAM trained on more than 1.5M image-mask pairs across multiple modalities. It established a strong promptable medical segmentation prior and made frozen-generalist / specialist combinations practical.

### SAM 2 / Medical SAM 2 (2024 onward)

SAM 2 added image/video memory and inspired methods that reinterpret volumetric medical images as sequences. Several 2025–2026 SSMIS papers now use SAM2/MedSAM2-family models.

## 4. Direct intersection: VFM + semi-supervised medical segmentation

This is the key literature cluster for the LEIA-style question.

### SemiSAM (BIBM 2024)

Uses SAM-assisted consistency regularization to improve specialist semi-supervised medical segmentation.

### SemiSAM+ (Medical Image Analysis 2025)

Formalises the **generalist foundation model + specialist SSL model** paradigm. The generalist can remain frozen; specialist predictions become prompts and generalist outputs provide additional supervision. Strongest gains are reported in very low-label regimes.

### SAMatch (2024/2025 line)

Combines SAM-guided pseudo-label refinement with match-style semi-supervised training.

### Stitching, Fine-Tuning, and Re-Training (TMI 2025)

A SAM-enabled semi-supervised framework for 3D medical segmentation, showing the general pattern extends beyond 2D prompt refinement.

### STAR (2025)

SAM-based teacher–student SSL plus contrastive consistency; uses parameter-efficient adaptation.

### CPAC-SAM (Medical Image Analysis 2026)

Cross prompting plus adaptive sampling and prompt consistency. Explicitly tackles unreliable prompts/pseudo-labels in unlabeled data.

### CPPS-SAM (JVCIR 2026)

Cross-prompting pseudo supervision between a SAM foundation model and a CNN expert.

### SSS / Semi-Supervised SAM-2 (2026)

SAM-2 based semi-supervised framework; reports ACDC low-label experiments. The public repository currently carries a caution about discrepancies found in some evaluation-script metrics, which is exactly why this benchmark should freeze and test its own evaluation code.

### UnCoL (TMI 2026)

Uncertainty-informed dual-teacher collaboration between generalist and specialist knowledge.

### ECT-3DMedSAM (MIDL 2026)

Efficient cross teaching using a 3D MedSAM-family foundation model.

### SemiSAM-O1 (arXiv 2026)

Pushes the annotation budget to a single labelled 3D volume. Uses the foundation encoder offline for feature-space label propagation and iterative uncertainty-guided refinement.

## 5. What the literature already implies

A minimal replication should not claim the following as novel:

- that SSL helps medical segmentation with few labels;
- that SAM/MedSAM can act as a generalist teacher;
- that pseudo-labels from a specialist can be refined by a foundation model;
- that uncertainty/confidence gating is useful;
- that gains are often largest under extreme label scarcity.

What *is* useful is an independent, tightly controlled answer to:

> **How much of the gain survives when we use the same split, same specialist, same training budget and an auditable evaluation protocol?**

That is the purpose of this repository.
