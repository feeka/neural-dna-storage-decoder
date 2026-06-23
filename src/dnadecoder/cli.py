"""Command-line interface for the ``dnadecoder`` package.

Sub-commands for synthetic data generation, training, evaluation, the full
experiment sweep, and a stand-alone Reed-Solomon outer-code demo. Wired up as the
``dnadecoder`` console script; ``main`` returns 0 on success.
"""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

import numpy as np

from .config import ChannelConfig, ModelConfig, TrainConfig


def _cmd_generate(args: argparse.Namespace) -> int:
    """Generate synthetic strands/reads and print a few samples + stats."""
    from . import channel as channel_mod

    cfg = ChannelConfig(
        length=args.length, num_traces=args.num_traces, p_sub=args.p_sub,
        p_del=args.p_del, p_ins=args.p_ins, source=args.source,
        markov_stay=args.markov_stay, seed=args.seed,
    )
    records = channel_mod.generate_from_config(cfg, args.num, seed=args.seed)
    print(f"Generated {len(records)} strands (source={cfg.source}, length={cfg.length}, "
          f"reads={cfg.num_traces}, p_sub={cfg.p_sub}).")
    for i in range(min(3, len(records))):
        rec = records[i]
        print(f"\n--- sample {i} ---")
        print(f"original: {rec['original']}")
        for j, tr in enumerate(rec["traces"]):
            print(f"  read[{j}] (len={len(tr)}): {tr}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Train a denoiser end to end and checkpoint it."""
    from . import train as train_mod

    channel_cfg = ChannelConfig(
        length=args.length, num_traces=args.num_traces, p_sub=args.p_sub,
        source=args.source, markov_stay=args.markov_stay, seed=args.seed,
    )
    model_cfg = ModelConfig(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        warmup_steps=args.warmup_steps, num_train=args.num_train,
        num_val=args.num_val, device=args.device, seed=args.seed,
        ckpt_path=args.ckpt_path,
    )
    _model, history = train_mod.build_and_train(channel_cfg, model_cfg, train_cfg)
    final = {k: v[-1] for k, v in history.items() if v}
    print("Training complete. Final metrics:")
    for k, v in final.items():
        print(f"  {k}: {v:.4f}")
    print(f"Checkpoint saved to {train_cfg.ckpt_path}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Load a checkpoint and compare the denoiser against baselines."""
    from . import channel as channel_mod
    from . import evaluate as evaluate_mod
    from . import train as train_mod

    model, meta = train_mod.load_checkpoint(args.ckpt, map_location=args.device)
    # Length and read count are fixed by the trained model.
    records = channel_mod.generate_dataset(
        args.num, meta["length"], meta["num_traces"], p_sub=args.p_sub,
        seed=args.seed, source=args.source, markov_stay=args.markov_stay,
    )
    results = evaluate_mod.compare(model, records, device=args.device)
    print(f"Evaluation at p_sub={args.p_sub} (source={args.source}, "
          f"length={meta['length']}, reads={meta['num_traces']}):")
    print(evaluate_mod.format_comparison_table(results))
    return 0


def _cmd_experiment(args: argparse.Namespace) -> int:
    """Run the full train-then-sweep experiment and print result tables."""
    from . import evaluate as evaluate_mod
    from . import experiment as experiment_mod

    out = experiment_mod.run_experiment(
        quick=args.quick, out_dir=args.out_dir, device=args.device
    )
    for p in sorted(out["results"]):
        print(f"\n## Substitution rate p = {p:g}")
        print(evaluate_mod.format_comparison_table(out["results"][p]))
    print(f"\nArtifacts written to: {out['out_dir']}")
    print(f"  - {out['out_dir']}/results.md")
    print(f"  - {out['out_dir']}/ser_vs_noise.png")
    return 0


def _cmd_rs_demo(args: argparse.Namespace) -> int:
    """Show the Reed-Solomon outer code correcting injected symbol errors."""
    from .outercode import ReedSolomon

    rng = np.random.default_rng(args.seed)
    rs = ReedSolomon(args.n, args.k)
    message = [int(x) for x in rng.integers(0, 256, size=args.k)]
    codeword = [int(x) for x in rs.encode(message)]

    max_err = (args.n - args.k) // 2
    received = list(codeword)
    for pos in rng.choice(len(received), size=max_err, replace=False):
        received[pos] = (received[pos] + int(rng.integers(1, 256))) % 256

    decoded_msg, n_corrected = rs.decode(received)
    decoded_msg = [int(x) for x in decoded_msg]
    print(f"ReedSolomon(n={args.n}, k={args.k}); corrupted {max_err} symbol(s)")
    print(f"  message : {message}")
    print(f"  codeword: {codeword}")
    print(f"  received: {received}")
    print(f"  decoded : {decoded_msg}  (corrected {n_corrected} error(s))")
    print(f"  recovered original message: {decoded_msg == message}")
    return 0


def _add_channel_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--length", type=int, default=ChannelConfig.length)
    p.add_argument("--num-traces", type=int, default=ChannelConfig.num_traces)
    p.add_argument("--p-sub", type=float, default=ChannelConfig.p_sub)
    p.add_argument("--source", type=str, default=ChannelConfig.source,
                   choices=["markov", "uniform"])
    p.add_argument("--markov-stay", type=float, default=ChannelConfig.markov_stay)
    p.add_argument("--seed", type=int, default=ChannelConfig.seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dnadecoder",
        description="Neural consensus decoder for noisy DNA sequencing reads.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate synthetic strands/reads")
    g.add_argument("--num", type=int, default=5)
    g.add_argument("--p-del", type=float, default=0.0)
    g.add_argument("--p-ins", type=float, default=0.0)
    _add_channel_flags(g)
    g.set_defaults(func=_cmd_generate)

    t = sub.add_parser("train", help="train the denoiser")
    _add_channel_flags(t)
    t.add_argument("--d-model", type=int, default=ModelConfig.d_model)
    t.add_argument("--nhead", type=int, default=ModelConfig.nhead)
    t.add_argument("--num-layers", type=int, default=ModelConfig.num_layers)
    t.add_argument("--dim-feedforward", type=int, default=ModelConfig.dim_feedforward)
    t.add_argument("--dropout", type=float, default=ModelConfig.dropout)
    t.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    t.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    t.add_argument("--lr", type=float, default=TrainConfig.lr)
    t.add_argument("--warmup-steps", type=int, default=TrainConfig.warmup_steps)
    t.add_argument("--num-train", type=int, default=TrainConfig.num_train)
    t.add_argument("--num-val", type=int, default=TrainConfig.num_val)
    t.add_argument("--device", type=str, default=TrainConfig.device)
    t.add_argument("--ckpt-path", type=str, default=TrainConfig.ckpt_path)
    t.set_defaults(func=_cmd_train)

    e = sub.add_parser("evaluate", help="evaluate a checkpoint vs baselines")
    e.add_argument("--ckpt", type=str, required=True)
    e.add_argument("--num", type=int, default=300)
    e.add_argument("--p-sub", type=float, default=ChannelConfig.p_sub)
    e.add_argument("--source", type=str, default=ChannelConfig.source,
                   choices=["markov", "uniform"])
    e.add_argument("--markov-stay", type=float, default=ChannelConfig.markov_stay)
    e.add_argument("--seed", type=int, default=123)
    e.add_argument("--device", type=str, default="cpu")
    e.set_defaults(func=_cmd_evaluate)

    x = sub.add_parser("experiment", help="run the full train+sweep experiment")
    x.add_argument("--quick", action="store_true")
    x.add_argument("--out-dir", type=str, default="results")
    x.add_argument("--device", type=str, default="cpu")
    x.set_defaults(func=_cmd_experiment)

    r = sub.add_parser("rs-demo", help="Reed-Solomon error-correction demo")
    r.add_argument("--n", type=int, default=12, help="total RS symbols")
    r.add_argument("--k", type=int, default=6, help="data RS symbols")
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(func=_cmd_rs_demo)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
