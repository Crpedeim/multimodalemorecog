# Dynamic Modality Weighting for Emotion Recognition

An end-to-end **multimodal emotion recognition** pipeline on **CMU-MOSEI**, fusing text, audio, and visual streams to classify sentiment — and a study of *why* a sophisticated cross-modal attention model can be beaten by a naive baseline.

**The headline finding:** a carefully engineered Cross-Modal Attention network with dynamic gating (**0.670 weighted F1**) was *outperformed* by simple feature concatenation (**0.673 weighted F1**). This repo documents that result and provides a gradient-level diagnosis of the failure mode — **"Modality Starvation."** 

---

## TL;DR

- Built three modality encoders and two fusion strategies in PyTorch, benchmarked against single-modality baselines on CMU-MOSEI.
- Multimodal fusion beat the strong text-only baseline (+2.78% F1) — confirming the modalities carry complementary signal.
- But **dynamic attention gating underperformed naive concatenation**, because the softmax gate collapsed almost entirely onto text.
- Diagnosed this via the chain rule: near-zero gates scale the backward gradients to the weaker encoders toward zero, inducing a localized vanishing-gradient that **starves the audio and visual branches** before they can learn.

---

## Architecture

```
Text  (N, 50, 768)  BERT embeddings   ─→  Bi-LSTM encoder            ─┐
Audio (N, 50, 74)   COVAREP features  ─→  Bi-LSTM encoder            ─┤→  128-d each
Video (N, 50, 35)   OpenFace AUs      ─→  Temporal Vision Transformer ─┘
                                                                        │
                                          ┌─────────────────────────────┴──────────────┐
                                          │   Fusion (two variants compared)            │
                                          │   A) Naive Concatenation → 384-d → MLP      │
                                          │   B) Cross-Modal Attention → softmax gate → │
                                          │      weighted sum Z → classifier            │
                                          └─────────────────────────────────────────────┘
```

**Encoders**
- **Text & Audio — Bidirectional LSTMs** capturing forward/backward temporal dependencies; final state is the concatenated forward + backward hidden vectors.
- **Video — Temporal Vision Transformer** over OpenFace facial-action-unit sequences: linear projection → sinusoidal positional encoding → multi-head self-attention → attention pooling to a single 128-d vector. (Chosen over an RNN to capture long-range dependencies in micro-expressions.)

**Fusion**
- **Naive Concatenation (baseline):** `[T ⊕ A ⊕ V]` → MLP classifier.
- **Cross-Modal Attention (main):** a differentiable softmax gate produces per-sample weights `(g_t, g_a, g_v)` that sum to 1; the fused representation is `Z = g_t·T + g_a·A + g_v·V`.

---

## Results (CMU-MOSEI test set)

| Model | Weighted F1 | Macro F1 | Accuracy |
|---|---|---|---|
| Audio only | 0.381 | 0.334 | 0.413 |
| Video only | 0.462 | 0.411 | 0.482 |
| Text only | 0.652 | 0.615 | 0.658 |
| **Naive Concatenation** | **0.673** | **0.638** | **0.675** |
| Cross-Modal Attention | 0.670 | 0.637 | 0.666 |

**Learned attention gate weights:** text **0.995**, audio **0.004**, video **0.001** — the gate effectively ignores two of the three modalities.

---

## The Modality Starvation phenomenon

Because text had far higher standalone predictive power (0.652 F1) than the from-scratch vision (0.462) and audio (0.381) encoders, the attention gate took the path of least resistance and drove the audio/video weights to ≈0.001 within the first few epochs.

The trap is in the backward pass. Since the fused output depends on the term `g_v · V_feat`, the gradient reaching the visual encoder is **scaled by `g_v`**. Multiplying incoming gradients by ~0.001 is an artificial, localized vanishing-gradient: the weak encoders receive almost no error signal, never learn, and stay weak — a self-reinforcing collapse.

**Why naive concatenation wins:** the MLP can't mute an entire modality with a single scalar, so optimization is *forced* to find usable signal in the audio/visual dimensions, preserving gradient flow to every branch.

Proposed fixes (in the paper): modality-specific dropout on the dominant text vector, gradient blending across modalities, and pre-training the vision encoder before it enters the gated fusion.

---

## Training setup

Adam (weight decay 1e-4) · **dual learning rate** — 1e-3 for the newly-initialized ViT and fusion blocks, 1e-5 for the unfrozen pre-trained text/audio encoders to avoid catastrophic forgetting · weighted cross-entropy for class imbalance · batch size 64 · up to 50 epochs with early stopping (patience 7 on val F1).

---

## Repo structure

```
models/        text, audio (Bi-LSTM), vision (Temporal ViT) encoders + fusion + full model
scripts/       train_mosei_baselines.py · train_fusion.py
data/          mosei_dataset.py (loading, alignment to T=50, NaN→zero handling)
configs/       mosei_config.py
results/        saved models + history/test JSON for all 5 configurations
```

## Run

```bash
pip install torch numpy scikit-learn
python scripts/train_mosei_baselines.py   # single-modality baselines
python scripts/train_fusion.py            # concat + attention fusion
```

*(Requires pre-extracted CMU-MOSEI features: BERT text, COVAREP audio, OpenFace visual.)*
