# Energy terms

Reference for every energy term in the Monte Carlo sequence-design objective,
with equations and normalization. Lower energy = better; MC minimizes.

## Notation

| Symbol | Meaning |
|---|---|
| `S`, `L` | candidate sequence (`s_i` ∈ {A,U,G,C}), its length |
| `idx(n)` | alphabet index (A=0, U=1, G=2, C=3) |
| `P` | target base-pair map (involution); `n_pairs = |P|/2` |
| `C` | DSSR contact list `{(i, j, lw)}`; `N_c = |C|` |
| `w_k` | weight of term `k` |

## Structure of the objective

Two levels. The top level is a weighted sum over three **composite blocks**; each
block is a weighted sum over its sub-terms.

```
E(S) = w_evo·E_evo + w_sec·E_sec + w_ter·E_ter        where  E_block = Σ_j a_j·E_j
ΔE(p→a) = same form, per-term deltas       (MC never recomputes the full sum)
```

| Block | Sub-terms |
|---|---|
| `evo` | evo |
| `secondary` | ss, stacking_context, ensemble_defect, junction |
| `tertiary` | hbond, sa, torsion |

**Gauge.** A `CompositeTerm`'s sub-weights `a_j` are simplex-normalized (Σ`a_j` = 1),
so a block carries no magnitude of its own: the `a_j` set the *mix* inside the
block, `w_block` sets the block's *magnitude*. Without this the two levels would
be a pure reparameterization of each other and the weights unidentifiable.

`ss_rnafold` is **not** in `default()` — an ablation
(`scripts/ablate_secondary_terms.py`) found it strongly redundant with
`ensemble_defect` (within-target Pearson ~0.8) at the same O(L³) cost. It
survives only in the `rna_design*` constructors. `rna_fm` is likewise available
but not wired into `default()`.

## Normalization principle

Terms are **intensive** (size-independent), so one weight transfers across
structures of different length. Sub-terms in the same block should share a
denominator, or the fixed `a_j` buys a target-dependent effective weight.

| Term | Divisor | Range |
|---|---|---|
| evo, sa, torsion, junction, rna_fm | `L` | unbounded |
| hbond | `L` | unbounded |
| ss, stacking_context | `n_pairs` | ss ∈ [0,1] |
| ensemble_defect | `L` | [0, 1] |
| ss_rnafold | none needed — `1 − MCC` is intensive | [0, 2] |

> **Known wart.** The `secondary` block is still mixed: `ss`/`stacking_context`
> are per-pair while `ensemble_defect`/`junction` are per-nucleotide, so their
> relative weighting drifts with each target's pair density. `tertiary` is
> uniformly per-nucleotide.

---

## `evo` — PSSM log-odds (`EvoTerm`)

Rewards evolutionarily conserved nucleotides from the structural-alignment PSSM
`M ∈ ℝ^{L×4}` (`M[i,n] = log( f_i(n) / bg(n) )`).

```
E_evo(S)    = −(1/L) Σ_i M[i, idx(s_i)]
ΔE_evo(p→a) = −( M[p, idx(a)] − M[p, idx(s_p)] ) / L
```

---

## Secondary-structure terms

### `ss` — base-pair validity (`SSTerm`)
Fraction of target pairs that are **not** valid Watson–Crick/wobble
(`{A·U, G·C, G·U}`). Geometry-level, O(1) delta.

```
E_ss(S) = (1/n_pairs) Σ_{(i,j)∈P, i<j}  𝟙[ {s_i, s_j} ∉ {AU, GC, GU} ]
```

### `stacking_context` — stacking-aware pair quality (`StackingContextTerm`)
Rewards strong pairs (GC=3 > AU=2 > GU=0.8) that sit in **contiguous helical
stacks**, via a score matrix `S_stk` with Gaussian-decayed stacking propagation
(`exp(−k²)` for a run of length `k`).

```
E_stack(S) = −(1/n_pairs) Σ_{(i,j)∈P, i<j}  S_stk[i, j]
```
`delta` rebuilds `S_stk` (O(L²)): a mutation perturbs neighbouring stacked pairs,
so the change is summed over all target pairs, not just `p`'s.

### `ensemble_defect` — thermodynamic ensemble defect (`EnsembleDefectTerm`)
Normalized ensemble defect (Zadeh et al., 2011): expected fraction of nucleotides
mis-paired vs. target across the Boltzmann ensemble, so lower = the target
dominates. `p_i(target_i)` is the ensemble probability that `i` adopts its target
pairing state. O(L³) per evaluation — this is what makes the `secondary` block's
delta O(L³).

```
E_ed(S) = ( L − Σ_i p_i(target_i) ) / L        ∈ [0, 1]
```

### `ss_rnafold` — RNAfold structure match (`RNAfoldTerm`)
Folds `S` with ViennaRNA RNAfold (O(L³) subprocess); `delta` is a full refold.

```
target given:  E(S) = 1 − MCC( MFE(S), target )     ∈ [0, 2]; 0 = exact, 1 = chance
no target:     E(S) = −MFE(S) / L                   [kcal/mol per nt]
```

MCC over the base-pair classification — every one of the `L(L−1)/2` candidate
position pairs is a binary instance, with the target as ground truth:

```
MCC = (TP·TN − FP·FN) / sqrt( (TP+FP)(TP+FN)(TN+FP)(TN+FN) )        0 if any marginal is 0
```

MCC is intensive by construction (∈ [-1,1] at any length), unlike the raw
`bp_distance` count this term used previously. It is also exactly the scMCC the
evaluation pipeline reports, so search objective and reported 2-D metric agree.

---

## Knowledge-based propensity terms

`sa`, `torsion`, `junction` all consult a table
`Φ[class, nuc] = log( P(nuc | class) / P(nuc) )` derived from the gRNAde **TRAIN**
split (`scripts/propensity/derive_propensities.py`). Uniform/empty table → 0
(well-defined no-op). O(1) delta.

```
E(S)      = −(1/L) Σ_i Φ[ class(i), idx(s_i) ]
ΔE(p→a)   = −( Φ[class(p), idx(a)] − Φ[class(p), idx(s_p)] ) / L

with, from counts N[c,n] and pseudocount α:
P(n|c) = (N[c,n]+α) / Σ_{n'}(N[c,n']+α)      P(n) = (Σ_c N[c,n]+α) / Σ_{c,n'}(N+α)
```

| Term | `class(i)` |
|---|---|
| `sa` | `sasa_class(SASA_i)`: buried (<30 Å²) / intermediate (30–90) / exposed (>90) |
| `torsion` | `chi_bin(χ_i)`: `chi_low` (χ < −140°) / `chi_high` |
| `junction` | `ss_element(i)` ∈ {other, stem, hairpin, internal, bulge, junction} (DSSR) |

---

## `hbond` — DSSR H-bond / LW-family propensity (`DSSRHbondTerm`)

Rewards bases that can form the hydrogen bonds the native geometry requires at
each DSSR contact, **including non-canonical (Leontis–Westhof) pairs and base
multiplets** the SS terms miss. Per-contact score `Ψ(lw, s_i, s_j)`; O(degree)
delta since geometry is fixed and only the presented atoms change.

```
E_hbond(S)    = −(1/L) Σ_{(i,j,lw)∈C}  Ψ(lw, s_i, s_j)
ΔE_hbond(p→a) = −(1/L) Σ_{contacts touching p}  ( Ψ_new − Ψ_old )
```

Normalized by `L`, not `N_c`: both are intensive, but a per-contact mean would
make the fixed simplex weight inside `tertiary` buy a different effective
magnitude on every target, scaled by that target's contact density. With `/L` the
block mix is target-independent and H-bond-rich backbones legitimately score
stronger.

**Backend 1 — chemistry capacity (default, training-free).** With edges
`(e1, e2) = edges(lw)` (W/H/S) and per-edge donor/acceptor counts:

```
Ψ_chem = min( don(s_i,e1), acc(s_j,e2) ) + min( acc(s_i,e1), don(s_j,e2) )
```
i.e. the number of donor↔acceptor H-bonds the two edges can satisfy (cWW G·C → 3,
A·U → 2). *Caveat:* a compatibility proxy, not a free-energy scale — "more bonds"
is not strictly "more stable" (stacking dominates duplex stability).

**Backend 2 — learned LW-family propensity.** Canonicalize
`(family, a, b) = canonical(lw, s_i, s_j)` over the 12 families
`{c,t}×{WW,WH,WS,HH,HS,SS}` (edges ordered W<H<S, bases ordered for symmetric
families, so swapped partners collapse to one cell). Derived TRAIN-only
(`scripts/propensity/derive_hbond_propensity.py`) with Bayesian shrinkage toward
the chemistry prior (strength `λ`):

```
π_f[a,b]   = ( cap_f[a,b] + 0.5 ) / Σ ( cap_f + 0.5 )          (chemistry prior)
P(a,b | f) = ( N_f[a,b] + λ·π_f[a,b] ) / Σ ( N_f + λ·π_f )
Ψ_learned  = T_f[a,b] = log( P(a,b | f) / P(a,b) )             (P(a,b) pooled over families)
```

Families/cells absent from `T` — and non-ACGU bases — fall back to `Ψ_chem`, so an
empty table reproduces the chemistry backend exactly.

> Note the two backends are on different scales (a 0–3 bond count vs. log-odds)
> and a partially-populated table mixes them within one sum.

---

## `rna_fm` — RNA-FM pseudo-likelihood (`RNAFMTerm`)

Masked-LM pseudo-log-likelihood from RNA-FM (requires torch + multimolecule).
Not in `default()`.

```
E_fm(S)    = −(1/L) Σ_i log P_FM( s_i | S_{−i} )
ΔE_fm(p→a) = ( log P_FM(s_p | S_{−p}) − log P_FM(a | S_{−p}) ) / L
```

---

## Constructors (`EnergyFunction`)

| Constructor | Blocks / terms |
|---|---|
| `default` | `evo` + `secondary` + `tertiary` (the three-composite objective) |
| `evo_only` | `evo` |
| `structure_only` | `secondary` + `tertiary` |
| `structural` | flat (uncomposited): evo, ss, stacking_context, sa, torsion, junction; optional hbond, ensemble_defect |
| `rna_design` | flat: evo, ss_rnafold |
| `rna_design_stacking` | flat: evo, ss_rnafold, stacking_context |

`score()` expands composites: block `b` reports its own key plus one `b.<sub>` key
per sub-term, each already scaled by `w_block`.
