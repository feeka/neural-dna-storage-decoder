# neural-dna-decoder

**A Transformer that denoises DNA-storage reads by *learning the source prior* — performing MAP-style decoding that beats classical majority-vote consensus, with the margin growing as the channel gets noisier.**

[![CI](https://github.com/REPLACE_ME/neural-dna-decoder/actions/workflows/ci.yml/badge.svg)](https://github.com/REPLACE_ME/neural-dna-decoder/actions)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

> Trains on **CPU in minutes** — no GPU required. All data is synthetic, so there is nothing to download.

---

## The idea

DNA is a dense, durable medium for archival storage: bits are encoded into the four nucleotides `A/C/G/T`, synthesized, and read back by sequencing. Reading is noisy and redundant — you observe several corrupted **reads** of each original strand and must reconstruct it.

The classical fix is **symbol-wise majority voting** across the aligned reads. But majority voting is *source-agnostic*: it treats every position independently and ignores the fact that real encoded DNA is **not** a uniform random string — it has structure (constrained codes, biological motifs, correlations). The information-theoretically optimal decoder is **MAP**: combine the channel evidence (the reads) **with the source prior**.

This project shows that a small **Transformer learns exactly that prior** and uses it to beat majority voting — and that the advantage **grows with the noise level**, precisely because the prior matters more when the reads are less reliable.

To make the comparison crisp, the source is an **order-1 Markov chain** over `ACGT` (a tunable, low-entropy prior) and the channel is **substitution noise** (reads stay aligned to the original). The repo also includes the general **insertion/deletion/substitution (IDS)** channel, an indel-aware baseline (BMA), and a **Reed–Solomon** outer code, for the broader DNA-storage picture.

## Results

![Symbol-error-rate vs substitution rate](docs/ser_vs_noise.png)

Symbol-error-rate (lower is better) on held-out strands. Markov source (`stay=0.75`), length 24, **only 3 reads** per strand; one model trained on the mixed noise range in **~8 minutes on a laptop CPU**:

| Substitution rate | **Neural (this work)** | Majority vote | BMA |
| --- | ---: | ---: | ---: |
| 0.05 | 0.67% | 0.63% | 15.21% |
| 0.15 | **3.85%** | 4.79% | 35.10% |
| 0.25 | **10.90%** | 12.86% | 46.01% |
| 0.35 | **19.01%** | 22.12% | 52.94% |

At `p = 0.05` the channel is easy and majority voting is already near-optimal, so the two tie. As the substitution rate climbs, the neural decoder's learned prior lets it pull steadily ahead — a **~14% relative error reduction** over majority voting at `p = 0.35`, and consistently higher exact-strand recovery. (BMA is the indel-oriented baseline, shown applied to the substitution channel for reference; it is meant for the harder insertion/deletion regime.)

## Architecture

```
reads  (K noisy copies, position-aligned)            per-position
                                                       base logits
  read 0:  A C G T ...  ┐   embed each base
  read 1:  A C G A ...  ├─► + position enc (over L)  ┌───────────────┐   mean over   ┌──────────┐
  read 2:  T C G T ...  ┘   + read-index enc (over K)│  Transformer  │──► K reads ──►│  Linear  │──► A/C/G/T
                            flatten to K·L tokens     │   Encoder     │   per pos     │  (→4)    │   per position
                                                      └───────────────┘               └──────────┘
                            self-attention sees, per position:
                              • the other reads at that position  → learned majority vote (evidence)
                              • neighbouring positions             → the source prior
```

This is **non-autoregressive**: every base is predicted in a single forward pass, so there is no exposure bias (the failure mode that cripples a seq2seq decoder here — see *Design notes*). Because the model receives the full set of reads at each position, it can always reproduce majority voting, and it improves on it by exploiting the prior.

## Install

```bash
git clone https://github.com/REPLACE_ME/neural-dna-decoder.git
cd neural-dna-decoder
python -m pip install -e .          # CPU PyTorch is sufficient
python -m pip install -e ".[dev]"   # + pytest, for development
```

## Usage

```bash
# 1) Inspect the channel — Markov-source strands and their noisy reads
dnadecoder generate --num 3 --length 24 --num-traces 3 --p-sub 0.15 --source markov

# 2) Full experiment: train one model, sweep the substitution rate, write table + plot
dnadecoder experiment --out-dir results     # ~minutes on CPU
dnadecoder experiment --quick               # <1 min smoke run (CI)

# 3) Train a checkpoint, then evaluate it against the baselines
dnadecoder train --epochs 12 --num-train 10000 --ckpt-path checkpoints/model.pt
dnadecoder evaluate --ckpt checkpoints/model.pt --p-sub 0.25 --num 300

# 4) Reed–Solomon outer code correcting injected symbol errors
dnadecoder rs-demo --n 12 --k 6
```

`experiment` writes `results/results.md` (per-noise tables) and `results/ser_vs_noise.png` (the plot above).

## How it works

- **Source** (`dnadecoder.channel`): `uniform` (i.i.d.) or `markov` (order-1 chain with tunable self-transition `stay`). Markov gives a learnable prior.
- **Channel** (`dnadecoder.channel`): the substitution channel keeps reads aligned (length-preserving); the general IDS channel (`corrupt`) adds insertions/deletions.
- **Data** (`dnadecoder.data`): each record becomes a `[K, L]` grid of read base-indices and an `[L]` target; ragged reads are padded.
- **Model** (`dnadecoder.model`): an encoder-only `DenoiserTransformer` over the `K·L` read/position grid with learned position and read-index encodings, mean-pooled over reads and classified per position.
- **Training** (`dnadecoder.train`): per-position cross-entropy, Adam, optional LR warmup (linear ramp → inverse-sqrt decay) for fast, stable pre-LayerNorm convergence.
- **Baselines** (`dnadecoder.baselines`): `majority_vote` (position-wise consensus) and `bma` (Bitwise Majority Alignment — the indel-aware classic).
- **Outer code** (`dnadecoder.outercode`): `GF(256)` arithmetic and a systematic `ReedSolomon(n, k)` codec (syndromes + Berlekamp–Massey + Chien + Forney) correcting up to `(n−k)/2` symbol errors.
- **Metrics** (`dnadecoder.metrics`): edit distance, symbol-error-rate, exact-match rate.

## Design notes (why this architecture)

A natural first attempt is a **seq2seq Transformer** that reads the concatenated traces and autoregressively emits the strand. On this problem it fails: under teacher forcing it learns to predict the next base mostly from its own (correct) prefix and the Markov prior, **largely ignoring the read evidence** — so at inference (free-running greedy decoding) it generates plausible-but-wrong strands, with error rates near the prior and *independent of the noise level*. The **non-autoregressive per-position** formulation removes both the prefix shortcut and the exposure bias: there is no prefix to lean on, so the model is forced to use the reads, and every position is decoded in one shot. This is the difference between a decoder that loses 5× to majority vote and one that beats it.

## Repository layout

```
src/dnadecoder/
  tokens.py        ACGT <-> index mapping
  config.py        Channel / Model / Train dataclasses
  channel.py       Markov/uniform sources + substitution & IDS channels
  data.py          read-grid Dataset, collate, DataLoader
  metrics.py       edit distance, symbol-error-rate, exact-match
  baselines.py     majority vote + Bitwise Majority Alignment
  model.py         DenoiserTransformer (encoder-only, per-position)
  train.py         training loop (+ LR warmup) and checkpointing
  evaluate.py      neural-vs-baseline comparison + markdown tables
  experiment.py    train -> substitution-rate sweep -> results.md + plot
  cli.py           dnadecoder generate|train|evaluate|experiment|rs-demo
  outercode/       GF(256) arithmetic + Reed-Solomon codec
scripts/run_experiment.py
tests/             unit tests for every module
```

## Reproducibility & testing

```bash
pytest -q                       # full unit-test suite
dnadecoder experiment --quick   # end-to-end smoke run
```

All data generation is seeded; CI runs the tests and the quick experiment on every push.

## Limitations & future work

- **Synthetic channel.** The error model is i.i.d.; real sequencing profiles are context- and technology-dependent. Plugging in an empirical error/source model is a natural extension.
- **Indels.** The neural benchmark targets the (aligned) substitution channel; extending the learned decoder to the full IDS channel — e.g. with a learned alignment or a CTC/transducer head — is open. BMA is included as the classical indel-aware reference.
- **Outer code integration.** Reed–Solomon is implemented and demonstrated standalone; an end-to-end `bits → RS → DNA → channel → neural decode → RS → bits` pipeline is future work.
- **Scale.** Strand length, read count, and model size are kept small for CPU-friendliness; the same code scales up on a GPU.

## Selected references

- G. M. Church, Y. Gao, S. Kosuri. *Next-Generation Digital Information Storage in DNA.* Science, 2012.
- N. Goldman et al. *Towards practical, high-capacity, low-maintenance information storage in synthesized DNA.* Nature, 2013.
- R. Heckel, G. Mikutis, R. N. Grass. *A characterization of the DNA data storage channel.* Scientific Reports, 2019.
- T. Batu, S. Kannan, S. Khanna, A. McGregor. *Reconstructing strings from random traces.* SODA, 2004. (Bitwise Majority Alignment)
- I. S. Reed, G. Solomon. *Polynomial codes over certain finite fields.* J. SIAM, 1960.

## License

MIT — see [LICENSE](LICENSE).
