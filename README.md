# neural-dna-decoder

**A Transformer that reconstructs a DNA sequence from several noisy reads.** It learns the sequence's structure and beats per-position majority voting, by a wider margin as the reads get noisier.

[![CI](https://github.com/REPLACE_ME/neural-dna-decoder/actions/workflows/ci.yml/badge.svg)](https://github.com/REPLACE_ME/neural-dna-decoder/actions)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

> Trains on **CPU in minutes** — no GPU required. All data is synthetic, so there is nothing to download.

---

## The idea

Sequencing a DNA molecule gives you several noisy copies (reads), not one. Reconstructing the true sequence from them is a consensus problem — common in genomics, metagenomics, and DNA data storage. The standard method is per-position majority voting across the aligned reads.

Majority voting decides each position independently, ignoring the rest of the sequence. But DNA isn't a random string: native and metagenomic sequences have strong structure (k-mer statistics, motifs, codon bias), and engineered DNA carries code constraints. The optimal decoder (MAP) combines the read evidence with a prior over likely sequences.

A small Transformer learns that prior and beats majority voting, and the gap widens as the error rate rises.

The source here is an order-1 Markov chain over `ACGT`; the channel is substitution noise, so reads stay aligned. The repo also includes the general insertion/deletion/substitution (IDS) channel, an indel-aware baseline (BMA), and a Reed–Solomon codec. See [Limitations](#limitations) for what's idealized.

## Results

![Symbol-error-rate vs substitution rate](docs/ser_vs_noise.png)

Symbol-error-rate (lower is better) on held-out strands. Markov source (`stay=0.75`), length 24, **only 3 reads** per strand (a deliberately low-coverage regime); one model trained on the mixed noise range in **~8 minutes on a laptop CPU**:

| Substitution rate | **Neural (this work)** | Majority vote | BMA |
| --- | ---: | ---: | ---: |
| 0.05 | 0.67% | 0.63% | 15.21% |
| 0.15 | **3.85%** | 4.79% | 35.10% |
| 0.25 | **10.90%** | 12.86% | 46.01% |
| 0.35 | **19.01%** | 22.12% | 52.94% |

Majority vote is the right baseline here — for substitution noise with aligned reads it's near-optimal. At `p = 0.05` the channel is easy and the two tie; as the error rate rises the learned prior pulls the neural decoder ahead: ~14% lower symbol-error-rate at `p = 0.35`, and higher exact-strand recovery throughout.

BMA's numbers look bad because it's built for insertions/deletions, not substitutions: it holds back reads that disagree, assuming they're frame-shifted, so a single flipped base desyncs a read for the rest of the strand. It's not a fair baseline on this channel — it's included because it's the classic indel-aware method (see [Limitations](#limitations)).

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

It's non-autoregressive: every base is predicted in one forward pass, so there's no exposure bias (which is what sinks a seq2seq decoder here — see [Design notes](#design-notes-why-this-architecture)). Since the model sees all reads at every position, it can reproduce majority voting, and improve on it using the prior.

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

## Limitations

The channel is synthetic and idealized:

- Errors are i.i.d. and symmetric. Real sequencing errors are context-dependent (homopolymers, GC content, position in read) and asymmetric.
- Substitution-only, so reads stay aligned (Illumina-like). No indels — this doesn't model nanopore. The IDS channel is in the code but the model isn't extended to it.
- The high end of the noise sweep is unrealistic. Real per-base error rates are ~0.1–2%; rates up to 0.35 just spread the methods apart on the plot. Near realistic rates the methods are close.
- The Markov source models structured DNA, which fits native and metagenomic sequences. Stored data is usually encoded to be near-random, where the prior — and the neural advantage — shrinks.
- `K = 3` reads is low coverage.
- BMA targets indels, so its numbers here aren't a fair comparison; majority vote is the right baseline for this channel.

## Future work

- Use an empirical, context-dependent error model with realistic rates.
- Extend the model to the IDS channel (learned alignment, or a CTC / transducer head) to handle insertions and deletions.
- Wire up an end-to-end `bits → Reed–Solomon → DNA → channel → neural decode → RS → bits` storage pipeline.
- Scale strand length, coverage, and model size on a GPU (same code, larger configs).

## Selected references

- R. Vaser, I. Sović, N. Nagarajan, M. Šikić. *Fast and accurate de novo genome assembly from long uncorrected reads (Racon).* Genome Research, 2017. (consensus / read polishing)
- R. R. Wick, L. M. Judd, K. E. Holt. *Performance of neural network basecalling tools for Oxford Nanopore sequencing.* Genome Biology, 2019. (neural sequence decoding)
- R. Heckel, G. Mikutis, R. N. Grass. *A characterization of the DNA data storage channel.* Scientific Reports, 2019.
- T. Batu, S. Kannan, S. Khanna, A. McGregor. *Reconstructing strings from random traces.* SODA, 2004. (Bitwise Majority Alignment)
- I. S. Reed, G. Solomon. *Polynomial codes over certain finite fields.* J. SIAM, 1960. (Reed–Solomon)
- G. M. Church, Y. Gao, S. Kosuri. *Next-Generation Digital Information Storage in DNA.* Science, 2012. (related application: DNA data storage)

## License

MIT — see [LICENSE](LICENSE).
