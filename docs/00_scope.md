# Scope and falsifiable question

The goal is **not** to invent a new method on day one. The goal is to establish an auditable baseline that can answer a narrow question before adding complexity.

## Core question

Given the same medical images and the same small human-labelled subset:

1. How much does a standard specialist model learn from labelled data alone?
2. How much does a classical SSL method gain from the unlabelled remainder?
3. Does a frozen medical foundation model add anything beyond that SSL gain?
4. At what annotation fraction does the foundation-model advantage become negligible?

## Conditions

Let `L_f` be the patient subset labelled at fraction `f` and `U_f` the remaining train patients.

### SUP

`U-Net(L_f)`

### MT

`MeanTeacher(U-Net, L_f + U_f)`

### MT + MedSAM

`MeanTeacher(U-Net, L_f + U_f) + frozen MedSAM guidance`

MedSAM is not allowed to rewrite ground truth. It can only provide a second opinion / refinement signal on pseudo-labels for unlabelled images.

## What would count as evidence?

For each fraction and seed, compare test-set paired results on exactly the same patients.

Primary contrast:

`Δ_VFM(f) = Dice(MT+MedSAM, f) - Dice(MT, f)`

Secondary contrast:

`Δ_SSL(f) = Dice(MT, f) - Dice(SUP, f)`

The interesting object is not a single best Dice score but the **shape of Δ across annotation budgets**.
