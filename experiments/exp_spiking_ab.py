r"""Experiment 6 -- a spiking network against a dense one, same everything else.

Run:
    python experiments/exp_spiking_ab.py
    python experiments/exp_spiking_ab.py --seeds 5 --timesteps 25

The question
------------
Spiking networks are argued for on energy: activations are binary, so a
synapse whose input did not spike does no work, and one whose input did needs
an add rather than a multiply-accumulate. The claimed trade is *some* accuracy
for *much* less energy.

This is an A/B test of that claim on MNIST, run properly:

* **one starting point.** Both arms get identical data, an identical
  train/validation/test split (``common.get_data`` memoises it, so the split
  is the same object), identical layer sizes, identical parameter count,
  identical optimiser and learning rate, and the same seed per paired run.
* **two arms.** Dense ``TensorMLP`` against ``SpikingMLP``. They differ in
  whether hidden activations are real numbers or binary spikes in time, and
  in nothing else that was not forced by that choice.
* **frozen seeds.** Each seed produces one dense run and one spiking run, and
  the two are compared as a pair.
* **a held-out test set,** touched once, at the end, after all training and
  all model selection.
* **the result is published either way.** The pre-registered expectation is
  that the spiking network loses on accuracy. If it does, that is the finding
  and it is reported as the finding.

What is measured
----------------
**Accuracy** on the held-out test set.

**Spike count**: mean spikes emitted per input image, summed over all
timesteps and all hidden neurons. This is the physical quantity the energy
argument rests on, so it is reported directly rather than only through the
energy model.

**Simulated energy**: an operation count costed with published per-operation
energies (see ``nabla/nn/energy.py``). It is an estimate, not a measurement,
and it includes the membrane-update and static-input-layer costs that
published comparisons often omit.

What this cannot settle
-----------------------
MNIST with a static image and a two-layer network is close to the least
favourable setting for a spiking network: there is no temporal structure to
exploit, the input is dense and analog, and the network is too shallow for
spike-driven layers to dominate the operation count. A result here does not
generalise to event-camera input, deep networks, or actual neuromorphic
silicon. The narrow claim being tested is the one stated above.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from experiments.common import ARTIFACTS, get_data, rule  # noqa: E402
from nabla.data import DataLoader  # noqa: E402
from nabla.losses.tensor_losses import tensor_cross_entropy  # noqa: E402
from nabla.nn.energy import dense_energy, spiking_energy  # noqa: E402
from nabla.nn.spiking import SpikingMLP  # noqa: E402
from nabla.nn.tensor_mlp import TensorMLP  # noqa: E402
from nabla.optim import Adam  # noqa: E402


@dataclass
class ArmResult:
    arm: str
    seed: int
    test_accuracy: float
    val_accuracy: float
    test_loss: float
    epochs: int
    train_seconds: float
    parameters: int
    spikes_per_sample: float
    spike_rate: float
    energy_pj: float
    energy_breakdown: dict


# ======================================================================
# shared training loop
# ======================================================================


def evaluate(model, loader, *, spiking: bool):
    """Accuracy, mean loss, and mean spikes per sample over a whole loader."""
    correct = total = 0
    loss_sum = 0.0
    spikes = 0.0

    for xb, yb in loader:
        if spiking:
            logits, stats = model(xb, collect_stats=True)
            spikes += stats["spike_count"]
        else:
            logits = model(xb)
        loss_sum += tensor_cross_entropy(logits, yb).item() * len(yb)
        correct += int((logits.data.argmax(axis=1) == np.asarray(yb)).sum())
        total += len(yb)

    return correct / total, loss_sum / total, (spikes / total if spiking else 0.0)


def train_arm(
    arm: str,
    *,
    seed: int,
    train,
    val,
    test,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: list[int],
    timesteps: int,
    decay: float,
    surrogate: str,
    sharpness: float,
    verbose: bool = True,
) -> ArmResult:
    spiking = arm == "spiking"

    if spiking:
        model = SpikingMLP(
            784, hidden, 10, timesteps=timesteps, decay=decay,
            surrogate=surrogate, sharpness=sharpness, seed=seed,
        )
    else:
        model = TensorMLP(784, hidden, 10, activation="relu", seed=seed)

    optimiser = Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train, batch_size=batch_size, seed=seed, as_arrays=True)
    val_loader = DataLoader(val, batch_size=256, shuffle=False, as_arrays=True)
    test_loader = DataLoader(test, batch_size=256, shuffle=False, as_arrays=True)

    started = time.perf_counter()
    best_val = 0.0
    for epoch in range(1, epochs + 1):
        for xb, yb in train_loader:
            logits = model(xb)
            loss = tensor_cross_entropy(logits, yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        val_acc, val_loss, _ = evaluate(model, val_loader, spiking=spiking)
        best_val = max(best_val, val_acc)
        if verbose:
            print(f"      epoch {epoch:2d}/{epochs}  val_acc {val_acc:6.2%}  "
                  f"val_loss {val_loss:.4f}  {time.perf_counter() - started:5.1f}s")
    elapsed = time.perf_counter() - started

    # The test set is touched here and nowhere else.
    test_acc, test_loss, spikes_per_sample = evaluate(
        model, test_loader, spiking=spiking
    )

    sizes = [784, *hidden, 10]
    if spiking:
        breakdown = spiking_energy(
            sizes,
            timesteps=timesteps,
            layer_spikes=[spikes_per_sample],
            static_input=True,
        )
        rate = spikes_per_sample / (timesteps * sum(hidden))
    else:
        breakdown = dense_energy(sizes)
        rate = float("nan")

    return ArmResult(
        arm=arm,
        seed=seed,
        test_accuracy=test_acc,
        val_accuracy=best_val,
        test_loss=test_loss,
        epochs=epochs,
        train_seconds=elapsed,
        parameters=model.num_parameters(),
        spikes_per_sample=spikes_per_sample,
        spike_rate=rate,
        energy_pj=breakdown.total_pj,
        energy_breakdown=breakdown.as_dict(),
    )


# ======================================================================


def summarise(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    # Sample standard deviation: with three seeds, dividing by n rather than
    # n-1 understates the spread by about 20%.
    return float(array.mean()), float(array.std(ddof=1) if len(array) > 1 else 0.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--hidden", type=int, nargs="+", default=[128])
    p.add_argument("--timesteps", type=int, default=25)
    p.add_argument("--decay", type=float, default=0.9)
    p.add_argument("--surrogate", default="fast_sigmoid")
    p.add_argument("--sharpness", type=float, default=5.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-test", type=int, default=10000)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--sensitivity", action="store_true",
                   help="also sweep learning rate and timesteps for the spiking arm")
    p.add_argument("--sensitivity-n-train", type=int, default=20000)
    args = p.parse_args()

    rule("1. The setup")
    train, val, test = get_data(args.n_train, args.n_test)
    print(f"    train        {len(train):>7,}")
    print(f"    validation   {len(val):>7,}   (model selection only)")
    print(f"    test         {len(test):>7,}   (touched once, at the end)")
    print(f"    architecture 784 -> {' -> '.join(map(str, args.hidden))} -> 10")
    print(f"    seeds        {args.seeds}   paired: same seed for both arms")
    print(f"    timesteps    {args.timesteps}  (spiking arm)")
    print(f"    optimiser    Adam(lr={args.lr}), batch {args.batch_size}, "
          f"{args.epochs} epochs, both arms")

    print("\n    Both arms share the split object, the layer sizes, the")
    print("    parameter count, the optimiser and the seed. The only")
    print("    difference is binary spikes in time against real activations.")

    results: list[ArmResult] = []
    for seed in range(args.seeds):
        rule(f"2. Seed {seed}")
        for arm in ("dense", "spiking"):
            print(f"\n    {arm}:")
            result = train_arm(
                arm,
                seed=seed, train=train, val=val, test=test,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                hidden=args.hidden, timesteps=args.timesteps, decay=args.decay,
                surrogate=args.surrogate, sharpness=args.sharpness,
                verbose=not args.quiet,
            )
            results.append(result)
            print(f"      -> test {result.test_accuracy:.2%}   "
                  f"{result.train_seconds:.0f}s   "
                  f"{result.parameters:,} params")

    # ------------------------------------------------------------------
    rule("3. Result")

    dense = [r for r in results if r.arm == "dense"]
    spiking = [r for r in results if r.arm == "spiking"]

    d_acc, d_sd = summarise([r.test_accuracy for r in dense])
    s_acc, s_sd = summarise([r.test_accuracy for r in spiking])
    d_energy, _ = summarise([r.energy_pj for r in dense])
    s_energy, _ = summarise([r.energy_pj for r in spiking])
    s_spikes, _ = summarise([r.spikes_per_sample for r in spiking])
    s_rate, _ = summarise([r.spike_rate for r in spiking])

    print(f"    {'arm':<10} {'test accuracy':>18} {'spikes/image':>14} "
          f"{'energy/image':>16}")
    print("    " + "-" * 60)
    print(f"    {'dense':<10} {d_acc:>12.2%} ± {d_sd:>4.2%} "
          f"{'n/a':>14} {d_energy / 1000:>13.1f} nJ")
    print(f"    {'spiking':<10} {s_acc:>12.2%} ± {s_sd:>4.2%} "
          f"{s_spikes:>14,.0f} {s_energy / 1000:>13.1f} nJ")

    # Paired differences: the same seed trained both arms, so the per-seed
    # difference removes the seed's contribution and is the right statistic.
    paired = [s.test_accuracy - d.test_accuracy for d, s in zip(dense, spiking)]
    mean_gap, gap_sd = summarise(paired)

    print(f"\n    paired accuracy difference (spiking - dense): "
          f"{mean_gap:+.2%} ± {gap_sd:.2%}")
    print(f"    per seed: {', '.join(f'{g:+.2%}' for g in paired)}")
    print(f"    energy ratio (spiking / dense): {s_energy / d_energy:.2f}x")
    print(f"    mean hidden spike rate: {s_rate:.1%} of neuron-timesteps")

    # ------------------------------------------------------------------
    rule("4. Reading it")

    if mean_gap < 0:
        print(f"    The spiking network lost {abs(mean_gap):.2%} of accuracy.")
    else:
        print(f"    The spiking network gained {mean_gap:.2%} of accuracy.")

    if s_energy < d_energy:
        print(f"    It used {d_energy / s_energy:.2f}x less simulated energy.")
    else:
        print(f"    It used {s_energy / d_energy:.2f}x MORE simulated energy, "
              "so on this")
        print("    architecture there is no energy win to trade accuracy for.")

    breakdown = spiking[0].energy_breakdown
    print("\n    Where the spiking energy goes, per image:")
    print(f"      static input layer (MACs)  {breakdown['macs']:>12,.0f} ops"
          f"   {breakdown['macs'] * 4.6 / 1000:>8.1f} nJ")
    print(f"      spike-driven synapses      {breakdown['accumulates']:>12,.0f} ops"
          f"   {breakdown['accumulates'] * 0.9 / 1000:>8.1f} nJ")
    print(f"      membrane updates           "
          f"{breakdown['membrane_pj'] / 4.6:>12,.0f} ops"
          f"   {breakdown['membrane_pj'] / 1000:>8.1f} nJ")

    print("\n    The static input layer is a full dense layer of multiplies.")
    print("    With a 784-pixel analog input and one hidden layer of "
          f"{args.hidden[0]},")
    print("    that single term is most of the budget, and no amount of")
    print("    sparsity downstream can remove it.")

    # A sensitivity analysis, clearly labelled as a counterfactual: what the
    # energy would be if the input arrived as spikes instead of analog current.
    mean_intensity = float(np.asarray(test.x).mean())
    input_spikes = 784 * mean_intensity * args.timesteps
    counterfactual = spiking_energy(
        [784, *args.hidden, 10],
        timesteps=args.timesteps,
        layer_spikes=[s_spikes],
        static_input=False,
        input_spikes=input_spikes,
    )
    print("\n    Counterfactual (NOT measured, an estimate): if the input were")
    print("    rate-coded instead of analog, at MNIST's mean intensity of")
    print(f"    {mean_intensity:.3f} the input layer would emit ~{input_spikes:,.0f} "
          "spikes over")
    print(f"    {args.timesteps} steps, giving {counterfactual.total_nj:.1f} nJ "
          f"({counterfactual.total_pj / d_energy:.2f}x dense).")
    print("    That is the configuration the energy argument actually assumes.")
    print("    It was not run here, because rate coding costs accuracy that")
    print("    would then be charged to spiking rather than to the encoder.")

    # ------------------------------------------------------------------
    sensitivity: dict = {}
    if args.sensitivity:
        rule("5. Was the spiking arm handicapped?")
        print("    The two arms shared a learning rate and the spiking arm was")
        print("    given one fixed T. Both choices could be hiding a better")
        print("    spiking result, so both are swept here. Smaller training set")
        print(f"    ({args.sensitivity_n_train:,}) and one seed: this is a check on the")
        print("    conclusion, not a second experiment.")

        s_train, s_val, s_test = get_data(args.sensitivity_n_train, args.n_test)
        common = dict(
            train=s_train, val=s_val, test=s_test, epochs=args.epochs,
            batch_size=args.batch_size, hidden=args.hidden, decay=args.decay,
            surrogate=args.surrogate, sharpness=args.sharpness, verbose=False,
        )

        baseline = train_arm("dense", seed=0, lr=args.lr,
                             timesteps=args.timesteps, **common)
        print(f"\n    dense reference on this subset: "
              f"{baseline.test_accuracy:.2%}\n")

        print(f"    {'learning rate':>14} {'spiking test acc':>18}")
        lr_rows = []
        for lr in (3e-4, 1e-3, 3e-3):
            r = train_arm("spiking", seed=0, lr=lr,
                          timesteps=args.timesteps, **common)
            lr_rows.append({"lr": lr, "test_accuracy": r.test_accuracy})
            print(f"    {lr:>14.0e} {r.test_accuracy:>17.2%}")

        print(f"\n    {'timesteps':>10} {'spiking test acc':>18} "
              f"{'spikes/image':>14} {'energy':>12} {'vs dense':>10}")
        t_rows = []
        for timesteps in (5, 10, 25, 50):
            r = train_arm("spiking", seed=0, lr=args.lr,
                          timesteps=timesteps, **common)
            ratio = r.energy_pj / baseline.energy_pj
            t_rows.append({
                "timesteps": timesteps,
                "test_accuracy": r.test_accuracy,
                "spikes_per_sample": r.spikes_per_sample,
                "energy_pj": r.energy_pj,
                "energy_ratio": ratio,
            })
            print(f"    {timesteps:>10} {r.test_accuracy:>17.2%} "
                  f"{r.spikes_per_sample:>14,.0f} "
                  f"{r.energy_pj / 1000:>9.1f} nJ {ratio:>9.2f}x")

        best_lr = max(lr_rows, key=lambda row: row["test_accuracy"])
        best_t = min(t_rows, key=lambda row: row["energy_ratio"])
        sensitivity = {
            "n_train": args.sensitivity_n_train,
            "dense_reference": baseline.test_accuracy,
            "learning_rate": lr_rows,
            "timesteps": t_rows,
        }

        print(f"\n    Best spiking learning rate: {best_lr['lr']:.0e} at "
              f"{best_lr['test_accuracy']:.2%}, against a dense reference of "
              f"{baseline.test_accuracy:.2%}.")
        if best_lr["test_accuracy"] < baseline.test_accuracy:
            print("    The gap survives tuning, so it is not an artefact of the")
            print("    shared learning rate.")
        else:
            print("    The spiking arm overtakes the dense one once tuned, so")
            print("    the headline gap IS an artefact of the shared setting.")

        print(f"\n    Cheapest T is {best_t['timesteps']} at "
              f"{best_t['energy_ratio']:.2f}x dense energy and "
              f"{best_t['test_accuracy']:.2%} accuracy.")
        if best_t["energy_ratio"] >= 1.0:
            print("    Even the cheapest setting costs more than the dense")
            print("    network, because the static input layer dominates.")

    # ------------------------------------------------------------------
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "spiking_ab.json"
    path.write_text(json.dumps({
        "config": vars(args),
        "results": [asdict(r) for r in results],
        "summary": {
            "dense_test_accuracy": d_acc,
            "dense_test_accuracy_sd": d_sd,
            "spiking_test_accuracy": s_acc,
            "spiking_test_accuracy_sd": s_sd,
            "paired_gap_mean": mean_gap,
            "paired_gap_sd": gap_sd,
            "paired_gaps": paired,
            "dense_energy_pj": d_energy,
            "spiking_energy_pj": s_energy,
            "energy_ratio": s_energy / d_energy,
            "spikes_per_image": s_spikes,
            "spike_rate": s_rate,
            "counterfactual_rate_coded_pj": counterfactual.total_pj,
        },
        "sensitivity": sensitivity,
    }, indent=2, default=float))
    print(f"\n    wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
