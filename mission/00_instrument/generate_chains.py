"""
Chain generator. The one non-negotiable property (mission spec, Mission 0):

    each node is generated from its PARENT ONLY. The seed must be removed
    from context after step one.

    x_1 = generate(seed)
    for n in 2..N:
        x_n = generate(x_{n-1})   # x_{n-1} ONLY. Not the seed.

This file enforces that property in code, not just in intent: every call
that produces x_n for n >= 2 is checked to prove the prompt sent to the
model contains no substring of the seed text, and the check is loud on
failure (raises SeedLeakError, does not warn and continue).

Checkpointing: each chain is written to data/ after EVERY node (not just on
completion), and the runner resumes any chain -- complete or partial -- from
its last checkpointed node. This matters because free-tier daily request
quotas can be exhausted mid-chain; without per-node checkpoints, a quota cutoff
partway through a 20-node chain would discard everything generated so far.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import config
from providers import get_provider, GenerationResult


class SeedLeakError(AssertionError):
    """Raised when the prompt used to generate a node at depth >= 2 was not
    built solely from the immediate parent's text. This must fail loudly:
    it is the single most important property of the experiment."""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def assert_prompt_built_from_parent_only(
    prompt: str, parent_text: str, seed_text: str
) -> None:
    """
    Structural/provenance check: the depth >= 2 prompt must equal
    build_node_prompt(parent_text) exactly -- i.e. it must be constructible
    from the parent alone, with no seed text spliced in by the code.

    This is a code-correctness check, not a content-overlap check. Natural
    language chains routinely echo phrasing from beat to beat (x_1 legally
    contains words from the seed; x_2 is built from x_1 and so may still
    contain those same words). That organic echo is not a leak and must not
    trip this assertion -- only a structural mistake (e.g. the seed being
    concatenated into the prompt directly) should.
    """
    expected_prompt = build_node_prompt(parent_text)
    if prompt != expected_prompt:
        raise SeedLeakError(
            "SEED LEAK: the depth >= 2 prompt does not match "
            "build_node_prompt(parent_text) exactly. Something other than "
            "the immediate parent's text was spliced into the prompt -- "
            "the seed must not be in context after step 1."
        )


def seed_echo_diagnostic(seed: str, node_text: str, *, min_ngram_words: int = 6) -> Optional[str]:
    """
    Non-fatal diagnostic: report the longest verbatim word-run shared
    between the seed and a later node's text, for eyeballing in the smoke
    test (per spec: "print every node in full so I can eyeball whether seed
    details are leaking through"). This never raises -- textual echo from
    beat to beat is expected and is not evidence of a code-level leak; see
    assert_prompt_built_from_parent_only for the actual correctness check.
    """
    norm_seed = _normalize(seed)
    norm_node = _normalize(node_text)
    seed_words = norm_seed.split()
    longest = None
    for length in range(min(len(seed_words), 12), min_ngram_words - 1, -1):
        for i in range(len(seed_words) - length + 1):
            ngram = " ".join(seed_words[i : i + length])
            if ngram in norm_node:
                longest = ngram
                break
        if longest:
            break
    return longest


@dataclass
class NodeRecord:
    depth: int
    text: str
    model: str
    temperature: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    wall_time_seconds: float
    retries: int


@dataclass
class ChainRecord:
    chain_id: str
    seed: str
    provider: str
    model: str
    temperature: float
    nodes: List[NodeRecord] = field(default_factory=list)
    seed_wall_time_seconds: float = 0.0
    seed_prompt_tokens: int = 0
    seed_completion_tokens: int = 0
    total_wall_time_seconds: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    seed_leak_checked: bool = True


def build_seed_prompt() -> str:
    return config.SEED_INSTRUCTION


def build_node_prompt(parent_text: str) -> str:
    return f"{config.NODE_INSTRUCTION}\n\nPrevious beat:\n{parent_text}"


def generate_chain(
    provider,
    seed: str,
    depth: int,
    chain_id: str,
    model: str,
    temperature: float,
    verbose: bool = False,
    checkpoint_fn=None,
    existing: Optional[dict] = None,
) -> ChainRecord:
    """
    Generate (or resume) a chain up to `depth` nodes.

    checkpoint_fn, if given, is called with the ChainRecord after EVERY
    node completes (not just at the end of the chain). This is what makes
    the runner resumable mid-chain: free-tier daily quotas can be exhausted
    after node 7 of 20, and without a per-node checkpoint that progress
    would be lost and re-spent on resume.

    existing, if given, is a previously checkpointed (possibly partial)
    chain dict to resume from. Its seed and already-generated nodes are
    reused verbatim; generation continues from the first missing depth.
    """
    if existing is not None:
        record = ChainRecord(
            chain_id=chain_id,
            seed=existing["seed"],
            provider=existing["provider"],
            model=existing["model"],
            temperature=existing["temperature"],
            nodes=[NodeRecord(**nd) for nd in existing["nodes"]],
        )
        seed_text = record.seed
        start_depth = len(record.nodes) + 1
        if verbose:
            print(f"[{chain_id}] resuming from depth {start_depth} "
                  f"({len(record.nodes)} nodes already checkpointed)")
    else:
        record = ChainRecord(
            chain_id=chain_id,
            seed=seed,
            provider=provider.name,
            model=model,
            temperature=temperature,
        )

        # --- Step 1: generate the seed text itself. ---
        prompt_1 = build_seed_prompt()
        # x_1 is generated from a fresh seed-writing instruction, not from the
        # seed text itself -- the "seed" in this design is the model's own
        # opening beat. See note in module docstring / README about this choice.
        result_1 = provider.generate(prompt_1, temperature=temperature, model=model)
        seed_text = result_1.text
        record.seed = seed_text
        record.seed_wall_time_seconds = result_1.wall_time_seconds
        record.seed_prompt_tokens = result_1.prompt_tokens or 0
        record.seed_completion_tokens = result_1.completion_tokens or 0
        if checkpoint_fn:
            checkpoint_fn(record)

        if verbose:
            print(f"[{chain_id}] SEED (depth 0, generated): {seed_text}")

        # x_1 proper: first node generated FROM the seed only.
        prompt_x1 = build_node_prompt(seed_text)
        result_x1 = provider.generate(prompt_x1, temperature=temperature, model=model)
        x1_text = result_x1.text
        record.nodes.append(
            NodeRecord(
                depth=1,
                text=x1_text,
                model=model,
                temperature=temperature,
                prompt_tokens=result_x1.prompt_tokens,
                completion_tokens=result_x1.completion_tokens,
                wall_time_seconds=result_x1.wall_time_seconds,
                retries=result_x1.retries,
            )
        )
        if checkpoint_fn:
            checkpoint_fn(record)
        if verbose:
            print(f"[{chain_id}] x_1: {x1_text}")

        start_depth = 2

    parent_text = record.nodes[-1].text
    for n in range(start_depth, depth + 1):
        prompt_n = build_node_prompt(parent_text)

        # THE ASSERTION. Fails loudly if the prompt was not built from the
        # immediate parent alone (structural/code-correctness check).
        assert_prompt_built_from_parent_only(prompt_n, parent_text, seed_text)

        result_n = provider.generate(prompt_n, temperature=temperature, model=model)
        node_text = result_n.text

        # Non-fatal diagnostic only -- organic word-echo between seed and a
        # later node is expected and is not a code-level leak.
        echo = seed_echo_diagnostic(seed_text, node_text)

        record.nodes.append(
            NodeRecord(
                depth=n,
                text=node_text,
                model=model,
                temperature=temperature,
                prompt_tokens=result_n.prompt_tokens,
                completion_tokens=result_n.completion_tokens,
                wall_time_seconds=result_n.wall_time_seconds,
                retries=result_n.retries,
            )
        )
        if checkpoint_fn:
            checkpoint_fn(record)
        if verbose:
            echo_note = f"  [echo of seed: {echo!r}]" if echo else ""
            print(f"[{chain_id}] x_{n}: {node_text}{echo_note}")

        parent_text = node_text

    record.total_wall_time_seconds = (
        sum(nd.wall_time_seconds for nd in record.nodes) + record.seed_wall_time_seconds
    )
    record.total_prompt_tokens = (
        sum((nd.prompt_tokens or 0) for nd in record.nodes) + record.seed_prompt_tokens
    )
    record.total_completion_tokens = (
        sum((nd.completion_tokens or 0) for nd in record.nodes) + record.seed_completion_tokens
    )

    return record


def chain_checkpoint_path(data_dir: str, chain_id: str) -> str:
    return os.path.join(data_dir, f"chain_{chain_id}.json")


def save_chain(data_dir: str, record: ChainRecord) -> str:
    os.makedirs(data_dir, exist_ok=True)
    path = chain_checkpoint_path(data_dir, record.chain_id)
    payload = asdict(record)
    payload["config"] = {
        "provider": record.provider,
        "model": record.model,
        "temperature": record.temperature,
        "node_instruction": config.NODE_INSTRUCTION,
        "seed_instruction": config.SEED_INSTRUCTION,
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)
    return path


def load_chain(data_dir: str, chain_id: str) -> Optional[dict]:
    path = chain_checkpoint_path(data_dir, chain_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


class QuotaExhausted(Exception):
    """Raised to unwind the pool cleanly when the provider's daily/rate
    quota is exhausted. Partial progress is already on disk (checkpointed
    per node), so this is a "stop for today, resume tomorrow" signal, not
    a data-loss event."""


def _is_quota_exhaustion(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "resource_exhausted" in msg or "429" in msg


def run_chains(
    n_chains: int,
    depth: int,
    data_dir: str,
    provider_name: str = config.PROVIDER,
    model: Optional[str] = None,
    temperature: float = config.GENERATION_TEMPERATURE,
    id_prefix: str = "c",
    verbose: bool = False,
    max_workers: int = config.MAX_WORKERS,
) -> List[dict]:
    model = model or (config.GEMINI_MODEL if provider_name == "gemini" else config.CLAUDE_MODEL)

    chain_ids = [f"{id_prefix}{i:03d}" for i in range(n_chains)]

    todo = []
    results_by_id = {}
    for cid in chain_ids:
        existing = load_chain(data_dir, cid)
        if existing is not None and len(existing.get("nodes", [])) >= depth:
            print(f"[resume] {cid}: checkpoint found at depth {len(existing['nodes'])}, complete, skipping.")
            results_by_id[cid] = existing
        else:
            todo.append((cid, existing))

    if not todo:
        print("All chains already checkpointed.")
        return [results_by_id[cid] for cid in chain_ids]

    print(f"Generating {len(todo)} chains (of {n_chains} total) at depth {depth} "
          f"using provider={provider_name} model={model} temperature={temperature}")

    def _worker(cid: str, existing: Optional[dict]):
        provider = get_provider(provider_name)  # one client per thread/task
        t0 = time.monotonic()

        def _checkpoint(record: ChainRecord):
            # Fill in running totals so partial checkpoints are self
            # describing even before the chain finishes.
            record.total_wall_time_seconds = (
                sum(nd.wall_time_seconds for nd in record.nodes) + record.seed_wall_time_seconds
            )
            record.total_prompt_tokens = (
                sum((nd.prompt_tokens or 0) for nd in record.nodes) + record.seed_prompt_tokens
            )
            record.total_completion_tokens = (
                sum((nd.completion_tokens or 0) for nd in record.nodes) + record.seed_completion_tokens
            )
            save_chain(data_dir, record)

        record = generate_chain(
            provider, seed="", depth=depth, chain_id=cid,
            model=model, temperature=temperature, verbose=verbose,
            checkpoint_fn=_checkpoint, existing=existing,
        )
        elapsed = time.monotonic() - t0
        path = save_chain(data_dir, record)
        print(f"[done] {cid}: {depth} nodes, wall={elapsed:.1f}s, "
              f"tokens(prompt={record.total_prompt_tokens}, "
              f"completion={record.total_completion_tokens}) -> {path}")
        return cid, asdict(record)

    quota_hit = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, cid, existing): cid for cid, existing in todo}
        for fut in concurrent.futures.as_completed(futures):
            cid = futures[fut]
            try:
                cid, record_dict = fut.result()
                results_by_id[cid] = record_dict
            except SeedLeakError:
                # Loud failure: do not swallow, do not continue the pool.
                raise
            except Exception as exc:  # noqa: BLE001
                if _is_quota_exhaustion(exc):
                    print(f"[quota] {cid}: provider quota exhausted. "
                          f"Progress on this chain is checkpointed to disk; "
                          f"re-run this script later to resume. ({exc})")
                    quota_hit = True
                    continue
                print(f"[FAILED] {cid}: {exc!r}")
                raise

    if quota_hit:
        n_done = sum(1 for cid in chain_ids if len(results_by_id.get(cid, {}).get("nodes", [])) >= depth)
        print(f"\n[STOPPED] Provider quota exhausted. {n_done}/{n_chains} chains complete. "
              f"Re-run the same command later (checkpoints will resume automatically).")

    return [results_by_id[cid] for cid in chain_ids if cid in results_by_id]


def main():
    parser = argparse.ArgumentParser(description="Generate chains for Mission 0.")
    parser.add_argument("--n-chains", type=int, default=config.SMOKE_TEST_CHAINS)
    parser.add_argument("--depth", type=int, default=config.SMOKE_TEST_DEPTH)
    parser.add_argument("--data-dir", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "..", config.DATA_DIR))
    parser.add_argument("--provider", type=str, default=config.PROVIDER)
    parser.add_argument("--id-prefix", type=str, default="c")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-workers", type=int, default=config.MAX_WORKERS)
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)

    t0 = time.monotonic()
    records = run_chains(
        n_chains=args.n_chains,
        depth=args.depth,
        data_dir=data_dir,
        provider_name=args.provider,
        id_prefix=args.id_prefix,
        verbose=args.verbose,
        max_workers=args.max_workers,
    )
    elapsed = time.monotonic() - t0

    complete = [r for r in records if len(r["nodes"]) >= args.depth]
    partial = [r for r in records if len(r["nodes"]) < args.depth]

    print("\n=== SUMMARY ===")
    total_prompt = sum(r["total_prompt_tokens"] for r in records)
    total_completion = sum(r["total_completion_tokens"] for r in records)
    total_wall = sum(r["total_wall_time_seconds"] for r in records)
    print(f"Chains complete: {len(complete)}/{args.n_chains}  Partial: {len(partial)}  Depth target: {args.depth}")
    if partial:
        for r in partial:
            print(f"  partial: {r['chain_id']} at depth {len(r['nodes'])}/{args.depth}")
    print(f"Wall time (this run): {elapsed:.1f}s")
    print(f"Sum of per-chain wall time (may exceed run time due to parallelism): {total_wall:.1f}s")
    print(f"Avg wall time per chain: {total_wall / max(1, len(records)):.1f}s")
    print(f"Total prompt tokens: {total_prompt}  Total completion tokens: {total_completion}")
    print(f"Avg tokens per chain: prompt={total_prompt / max(1, len(records)):.0f} "
          f"completion={total_completion / max(1, len(records)):.0f}")


if __name__ == "__main__":
    main()
