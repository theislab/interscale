# Scientific background

This document summarizes the biological motivation, model design, and terminology from the
InterScale preprint so that the *purpose* behind the code — not just its structure — is clear.
Read this when working on anything touching model architecture, loss functions, the
interpretability/evaluation pipeline, or when the code's intent isn't obvious from names alone.

> Drummer, F., Jiménez, S., Di Marco, F., Schaar, A.C., Pentimalli, T.M., Beckman, J.L., Rajewsky, N.,
> Theis, F.J. *InterScale reveals multi-scale cellular interaction programs in spatial
> transcriptomics.* bioRxiv (2026). https://doi.org/10.64898/2026.05.07.723456
> (see also `docs/references.bib`, key `DrummerJimenez_2026`)

## Motivation

Cell–cell communication happens at multiple spatial scales simultaneously: autocrine/juxtacrine
signaling within a few micrometers, paracrine gradients across a local neighborhood, and
tissue-wide (or morphogen-gradient) coordination. Existing computational methods for inferring
cell-cell communication from spatial transcriptomics typically model *one* scale — either
adjacency-based local neighborhoods (GNN/niche methods like NCEM, NicheCompass) or generic
long-range dependencies — but not both jointly, and many rely on ligand-receptor priors that are
incomplete or infeasible for gene-panel-limited imaging-based platforms.

InterScale's core idea: combine a graph neural network (local, short-range, adjacency-constrained)
with a transformer (global, long-range, effectively fully-connected) in one architecture, so a
cell's transcriptional state is decomposed into local-neighborhood and tissue-wide contributions
that can be interpreted *separately*. This is why the codebase is organized around parallel
"local component" / "global component" module hierarchies rather than a single monolithic model.

## Architecture, mapped to code

The model follows the `GraphTrans` architecture (local GNN → global transformer), with these
building blocks:

- **Local component** (`interscale.module.local_modules`) — produces `H_local`, a per-cell
  embedding informed only by `k`-hop spatial neighbors (`k` = number of GCN/GIN layers). Four
  variants are implemented:
  - `GCN` — aggregates neighbor expression via the (self-loop-augmented, row-normalized) adjacency
    matrix; fixed at 2 layers to avoid oversmoothing/oversquashing.
  - `GIN` — Graph Isomorphism Network update rule with a learnable epsilon and an MLP.
  - `SCVI` — wraps `scvi.nn.Encoder`; captures gene-expression structure, *not* spatial
    neighborhood information (despite living in the "local" module tree).
  - `Precomputed` — loads embeddings computed outside InterScale (e.g. CellCharter, BANKSY) from
    `adata.obsm`; not trainable, so a dual-decoder setup isn't possible with this option.
- **From local to global**: `H_local` is LayerNorm'd, then padded/truncated to a fixed sequence
  length `S` per sliding window (`pad_batch`), and a CLS token is prepended. No positional
  encoding is added — the GNN already encoded spatial structure, and the transformer is meant to
  find interactions *without* distance restriction.
- **Global component** (`interscale.module.global_modules`,
  `TransformerNodeEncoderHook`) — a BERT-style multi-head self-attention transformer producing
  `H_global`, i.e. tissue/window-wide context. Notably, the attention mask is the *inverse*
  adjacency matrix (`M = 1 - A`): the transformer is explicitly steered toward attending to cells
  that are **not** already connected in the local graph, so local and global signal don't
  duplicate each other.
- **Decoders** (`interscale.nn`) — separate linear (or nonlinear) decoders for `H_local` and
  `H_global` reconstruct/predict gene expression or labels independently. Keeping the decoders
  separate is deliberate: it lets downstream analysis attribute an effect to "local" or "global"
  origin. The linear decoder is used for all reported experiments because it stays interpretable
  (decoder weights → standardized gene loadings, see below); the nonlinear decoder trades that
  away for accuracy.
- **Masking / self-supervision** — a fraction `pct_mask_nodes` of cells per graph have expression
  zeroed out (`tl.masking.apply_mask`); loss is computed only on masked cells. Classification uses
  weighted cross-entropy; regression uses a scaled cosine error loss (captures direction +
  magnitude) or Gaussian NLL.
- **`CombinedModule` vs `DualDecoderCombinedModule`** — the latter is what gives each component
  its own decoder (`cfg.model.decoder.dual_decoder = True`); this is required for the
  local-vs-global attribution analyses described below.

## Downstream interpretability — the actual point of the model

Prediction accuracy is a *proxy* used to confirm that local/global information is real signal, not
the end goal. The paper's real contribution is post-hoc, scale-resolved interpretation, at three
levels (`interscale.evaluation`):

- **Tissue/graph level** — the CLS token (connected to every cell) acts as a tissue-level summary.
  Its attention scores are decomposed into "vertical"/"horizontal" (sender/receiver) components and
  can be projected back onto the spatial slide to show which regions/cell types drive a
  classification (e.g. condition prediction).
- **Cell level** — raw attention weights aren't directly interpretable (attention-sink effects,
  softmax normalization differs per sliding window, layer/head averaging can wash out signal). The
  paper uses gradient-based **attention relevance** (averaging heads weighted by gradient
  importance, clamped ≥0, renormalized) rather than raw attention, and focuses on the **net
  attention flow** (incoming − outgoing, normalized per window) to get directional sender→receiver
  interaction strength between cell types, aggregated as a "flow map" and decomposed spatially into
  divergence-based **source/sink domains** via K-means.
- **Gene level** — decoder weights are converted into **standardized gene loadings** (decoder
  weight scaled by embedding-dimension-stdev / gene-stdev ratio) — this is what makes the linear
  decoder choice matter: it's a scale-independent, cell-type-agnostic measure of "how much does
  this embedding dimension drive this gene's expression," usable to rank informative embedding
  dimensions and run functional enrichment separately for local vs. global programs. A
  complementary approach compares each decoder's per-gene reconstruction rank (local rank − global
  rank) to double-check which scale "owns" a gene.
- **Moran's I across neighborhood size** is used throughout as an independent, model-free way to
  quantify a gene's effective spatial length scale (fast autocorrelation decay = local; slow decay
  = global) — used to validate that what the local/global embeddings pick up biologically matches
  genes' actual spatial organization.

## Key terminology / notation (paper → code)

| Paper notation | Meaning | Code |
|---|---|---|
| `X ∈ R^{N,F}` | gene expression matrix (cells × genes) | expression layer registered via `AnnDataManager` |
| `A` | adjacency / spatial connectivity matrix | built via `squidpy.gr.spatial_neighbors`, converted to edge index by `geome.transforms.AddEdgeIndex` |
| `H_local` / `E_local` | local (neighborhood) embedding | local module output |
| `H_global` / `E_global` | global (tissue-wide) embedding | global module (transformer) output |
| `CLS` | classification/tissue-summary token, attends to all cells | prepended before transformer; extracted in `CombinedModel.get_model_output` |
| `M = 1 - A` | transformer attention mask (blocks locally-connected pairs) | global module attention mask |
| `k` / `L_GCN` | number of GCN/GIN layers = local receptive field radius | `cfg.model.local_component.parameters.*` |
| `S` | max transformer sequence length per sliding window | `pad_batch` / dataloader |
| `p_m` / `pct_mask_nodes` | fraction of nodes masked per graph | `cfg.dataset` / `GraphAnnDataModule` |
| `M_net` | net attention flow matrix (directional sender→receiver) | attention relevance / net-flow analysis in `interscale.evaluation` |
| sliding window `w`, overlap `o`, step `T = w - o` | tissue partitioning for scalability | `prepare_geome_dataset` / squidpy sliding-window util |

## Datasets used in the paper (for context, not bundled with the repo)

- **Molecular Cartography SHH organoids** (Legnini et al.) — optogenetically induced Sonic
  Hedgehog signaling in neural tube organoids; this is the dataset behind `datasets._legnini` and
  `config_files/legnini_example.yaml`. Used to show InterScale recovers spatially-local neuronal
  differentiation programs (`GLI1`, `ISL1`, `TUBB3`) vs. broader progenitor/morphogen-regulation
  programs (`PROM1`, `NEUROG2`, `HHIP`) from the *same* tissue, without cell-type labels.
- **CosMx human pancreas (ND vs. T1D)** (Melton, Jiménez et al.) — used for node classification and
  the sender-receiver attention analysis; shows disease-associated reorganization (mast cell
  infiltration into islets in T1D) and separates local cell-state programs (endocrine/metabolic,
  e.g. `INS`, oxidative stress) from global immune/stromal signaling programs (`C1QC`, `IL32`,
  PI3K–AKT, focal adhesion).
- **10X Visium human brain, AD vs. control** (Chen et al., 2022) — used for graph-level
  classification benchmarking (local-only vs. global-only vs. combined), motivating why the global
  component matters even for "simple" classification tasks.
- **IMC pancreas T1D progression** (Damond et al., 2019) — robustness/generalization check.

## Known limitations (relevant when extending the model)

- Sliding windows cap the transformer's context length for scalability, so interactions spanning
  window boundaries are not modeled — this is a real ceiling on the "global" scale, not just an
  implementation detail (true long-range/endocrine signaling across whole organs is out of scope).
- No causal/directional inference between cell groups (e.g. A → C → B chains) — attention flow is
  correlational, not causal.
- No ground-truth for cell-cell communication exists, so validation is via proxy classification
  tasks and consistency with known biology (e.g. Moran's I, known marker genes), not direct
  benchmarking of inferred interactions.
- Raw attention weights are intentionally *not* used for interpretation (attention-sink effects,
  window-dependent softmax normalization) — always go through the relevance/net-flow pipeline in
  `interscale.evaluation`, not raw `attn_output_weights`.
