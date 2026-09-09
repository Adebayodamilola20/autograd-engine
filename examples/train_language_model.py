r"""Train a language model on your own text, then make it write.

Run it with no arguments for a built-in corpus, or point it at a file:

    python examples/train_language_model.py --text mybook.txt --steps 2000

What is actually happening
--------------------------
The model is trained on exactly one task: given the tokens so far, predict the
next one. That is the whole objective. Everything a language model appears to
know is a side effect of getting good at it, because predicting the next token
of real text well enough eventually requires representing what the text is
about.

Training samples a random window of ``block_size + 1`` tokens, feeds the first
``block_size`` in, and asks for the same window shifted one step left as the
target. Because the model is causal, one pass over a window of ``T`` tokens
yields ``T`` separate supervised predictions rather than one, which is what
makes this affordable at all.

Reading the loss
----------------
Cross-entropy here is in nats. The number to compare against is
:math:`\ln(V)`, the loss of a model that has learned nothing and guesses
uniformly over a vocabulary of size ``V``. Anything at or above that line
means no learning has happened yet.

The friendlier form is perplexity, :math:`e^{\text{loss}}`, which is roughly
"how many tokens is the model effectively choosing between". Perplexity 1 is
certainty; perplexity ``V`` is a coin toss over the whole vocabulary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from nabla.data.tokenizer import BPETokenizer, CharTokenizer  # noqa: E402
from nabla.losses.tensor_losses import tensor_cross_entropy  # noqa: E402
from nabla.nn.transformer import GPT  # noqa: E402
from nabla.optim import AdamW  # noqa: E402
from nabla.training.checkpoint import save_checkpoint  # noqa: E402

CORPUS = """All the world's a stage, and all the men and women merely players.
They have their exits and their entrances, and one man in his time plays many
parts, his acts being seven ages. To be, or not to be, that is the question:
whether 'tis nobler in the mind to suffer the slings and arrows of outrageous
fortune, or to take arms against a sea of troubles, and by opposing end them.
Now is the winter of our discontent made glorious summer by this sun of York,
and all the clouds that lour'd upon our house in the deep bosom of the ocean
buried. Friends, Romans, countrymen, lend me your ears; I come to bury Caesar,
not to praise him. The evil that men do lives after them; the good is oft
interred with their bones. What is past is prologue, and what to come is yours
and my discharge. We are such stuff as dreams are made on, and our little life
is rounded with a sleep. """


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * 74)


def batches(data: np.ndarray, block_size: int, batch_size: int, rng):
    """Sample ``batch_size`` random windows, and the same windows shifted by one.

    Random offsets rather than a sequential sweep: consecutive windows of a
    document are highly correlated, and a batch of correlated examples gives a
    gradient no better than a single one.
    """
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[s:s + block_size] for s in starts])
    y = np.stack([data[s + 1:s + block_size + 1] for s in starts])
    return x, y


def evaluate(model, data, block_size, batch_size, rng, batches_to_run=8):
    """Mean loss on held-out text. No backward pass, so no graph is kept."""
    total = 0.0
    for _ in range(batches_to_run):
        x, y = batches(data, block_size, batch_size, rng)
        logits = model(x)
        total += tensor_cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
        ).item()
    return total / batches_to_run


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--text", type=Path, help="a plain text file to learn from")
    p.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    p.add_argument("--vocab-size", type=int, default=512,
                   help="BPE only: ceiling on the learned vocabulary")
    p.add_argument("--block-size", type=int, default=64, help="context window")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=1500)
    # 1e-3, not the 3e-3 that works for the small MLPs elsewhere in this
    # repository. Measured on tinyshakespeare, 4 layers, 700 steps:
    #
    #     3e-3 -> val 2.527      1e-3 -> val 2.237      5e-4 -> val 2.267
    #
    # 3e-3 plateaus around 2.45 and stays there: at this depth the steps are
    # large enough to bounce across the minimum rather than settle into it.
    # 1e-3 passes that plateau in a third of the steps.
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=Path, help="write the trained model here")
    p.add_argument("--prompt", default="The ", help="seed text for the sample")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- data ---------------------------------------------------------
    rule("1. The text")
    if args.text:
        text = args.text.read_text(encoding="utf-8", errors="replace")
        print(f"  read {args.text} ({len(text):,} characters)")
    else:
        text = CORPUS * 12
        print(f"  no --text given, using a built-in corpus ({len(text):,} characters)")

    if args.tokenizer == "bpe":
        print(f"  training byte-level BPE up to {args.vocab_size} tokens ...")
        tokenizer = BPETokenizer.train(text, vocab_size=args.vocab_size)
    else:
        tokenizer = CharTokenizer.from_text(text)

    data = np.asarray(tokenizer.encode(text), dtype=np.intp)
    vocab_size = tokenizer.vocab_size
    print(f"  tokenizer: {args.tokenizer}, vocabulary {vocab_size:,}")
    print(f"  {len(data):,} tokens  ({len(text) / max(len(data), 1):.2f} chars/token)")

    # A held-out tail, so the reported loss is not just memorisation. The
    # split is by position rather than at random: shuffling would put nearly
    # identical overlapping windows on both sides and make validation
    # meaningless.
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    need = args.block_size + 2
    if len(val_data) < need:
        raise SystemExit(
            f"not enough text: the validation split holds {len(val_data)} tokens "
            f"but block_size={args.block_size} needs at least {need}. "
            "Use a longer file or a smaller --block-size."
        )
    print(f"  train {len(train_data):,} tokens   validation {len(val_data):,}")

    # --- model --------------------------------------------------------
    rule("2. The model")
    model = GPT(
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        seed=args.seed,
    )
    print("  " + model.summary().replace("\n", "\n  "))

    # --- training -----------------------------------------------------
    rule("3. Training")
    chance = float(np.log(vocab_size))
    print(f"  a model that learned nothing would score {chance:.4f} "
          f"(= ln {vocab_size}, perplexity {vocab_size})\n")

    optimiser = AdamW(model.parameters(), lr=args.lr)
    report_every = max(1, args.steps // 12)
    start = time.perf_counter()
    first_loss = None

    for step in range(1, args.steps + 1):
        x, y = batches(train_data, args.block_size, args.batch_size, rng)
        logits = model(x)
        loss = tensor_cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
        )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if first_loss is None:
            first_loss = loss.item()
        if step % report_every == 0 or step == args.steps:
            elapsed = time.perf_counter() - start
            val = evaluate(model, val_data, args.block_size, args.batch_size, rng)
            tokens = step * args.batch_size * args.block_size
            print(f"  step {step:5d}/{args.steps}  loss {loss.item():.4f}  "
                  f"val {val:.4f}  ppl {np.exp(val):7.2f}  "
                  f"{tokens / elapsed:8,.0f} tok/s")

    elapsed = time.perf_counter() - start
    final_val = evaluate(model, val_data, args.block_size, args.batch_size, rng)

    # --- sample -------------------------------------------------------
    rule("4. Writing")
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [int(train_data[0])]
    generated = model.generate(
        prompt_ids, 300, temperature=0.8, top_k=20, rng=rng
    )
    print(f"  prompt: {args.prompt!r}\n")
    print("  " + tokenizer.decode(generated).replace("\n", "\n  "))

    # --- result -------------------------------------------------------
    rule("Result")
    print(f"    parameters              {model.num_parameters():,}")
    print(f"    starting loss           {first_loss:.4f}")
    print(f"    final validation loss   {final_val:.4f}")
    print(f"    perplexity              {np.exp(final_val):.2f}  (chance: {vocab_size})")
    print(f"    trained in              {elapsed:.1f}s")

    print()
    if final_val >= chance - 0.05:
        print("    The loss has not moved below chance. Train for more steps, or")
        print("    raise --lr: at this point the model has learned nothing yet.")
    elif np.exp(final_val) < 2.0 and len(train_data) < 50_000:
        print("    Perplexity near 1 on a corpus this small means the model has")
        print("    memorised the text rather than learned it. Feed it more text")
        print("    with --text to see the difference.")
    else:
        print("    The model beats chance and is generating structured text.")
        print("    More text and more --steps is what improves it from here.")

    if args.save:
        path = save_checkpoint(
            args.save,
            model=model,
            epoch=args.steps,
            metadata={
                "kind": "gpt",
                "config": model.config(),
                "tokenizer": tokenizer.to_dict(),
                "val_loss": float(final_val),
            },
        )
        print(f"\n    saved to {path}")
        print(f"      nabla generate {path} --prompt \"{args.prompt}\"")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
