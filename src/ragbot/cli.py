"""Command line interface.

    ragbot ask  --corpus corpus --question "How long do refunds take?"
    ragbot chat --corpus corpus
    ragbot eval --corpus corpus --goldens goldens/ragbot.jsonl --set-baseline
"""
from __future__ import annotations

import argparse
import sys

from .chat import RagChatbot
from .embeddings import HashingEmbedder
from .llm import EchoContextLLM
from .retriever import HybridRetriever
from .tools import ToolRegistry, make_calculator_tool


def build_bot(corpus: str, backend: str = "echo", k: int = 4) -> RagChatbot:
    embedder = HashingEmbedder(dim=512)
    if backend == "openai":
        from .llm import OpenAILLM

        llm = OpenAILLM()
    elif backend == "anthropic":
        from .llm import AnthropicLLM

        llm = AnthropicLLM()
    else:
        llm = EchoContextLLM()

    bot = RagChatbot(
        llm=llm,
        embedder=embedder,
        retriever=HybridRetriever(embedder, k=k),
        tools=ToolRegistry([make_calculator_tool()]),
    )
    n = bot.ingest_path(corpus)
    print(f"indexed {n} chunks from {corpus}\n")
    return bot


def _show(answer) -> None:
    print(answer.text)
    if answer.citations:
        print("\nsources:")
        for c in answer.citations:
            print(f"  {c.marker} {c.source}  (score {c.score:.3f})")
    if answer.tool_calls:
        print("\ntools:")
        for t in answer.tool_calls:
            print(f"  {t['name']}({t.get('arguments')}) -> {t['result']}")
    print(f"\n[{answer.latency_ms:.0f}ms, grounded={answer.grounded}]")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ragbot", description="Grounded RAG chatbot")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--corpus", default="corpus")
        sp.add_argument("--backend", default="echo", choices=["echo", "openai", "anthropic"])
        sp.add_argument("-k", type=int, default=4, help="chunks to retrieve")

    ask = sub.add_parser("ask", help="ask a single question")
    common(ask)
    ask.add_argument("--question", required=True)

    chat = sub.add_parser("chat", help="interactive multi-turn session")
    common(chat)

    ev = sub.add_parser("eval", help="run the agenteval regression gate")
    common(ev)
    ev.add_argument("--goldens", default="goldens/ragbot.jsonl")
    ev.add_argument("--runs-dir", default="runs")
    ev.add_argument("--set-baseline", action="store_true")
    ev.add_argument("--tolerance", type=float, default=0.0)

    args = p.parse_args(argv)

    if args.cmd == "ask":
        _show(build_bot(args.corpus, args.backend, args.k).chat(args.question))
        return 0

    if args.cmd == "chat":
        bot = build_bot(args.corpus, args.backend, args.k)
        print("type 'exit' to quit, 'reset' to clear history\n")
        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if q.lower() in {"exit", "quit"}:
                return 0
            if q.lower() == "reset":
                bot.reset()
                print("(history cleared)\n")
                continue
            if not q:
                continue
            print()
            _show(bot.chat(q))
            print()

    # eval
    try:
        from agenteval.evaluator import Evaluator, load_goldens
        from agenteval.regression import compare
        from agenteval.store import RunStore
    except ImportError:
        print("agenteval not installed: pip install -e ../agent-eval-harness", file=sys.stderr)
        return 2

    from .eval_adapter import RagbotAgent, default_graders

    bot = build_bot(args.corpus, args.backend, args.k)
    agent = RagbotAgent(bot)
    cases = load_goldens(args.goldens)
    run = Evaluator(agent, graders=default_graders(), max_workers=1).run(cases)
    store = RunStore(args.runs_dir)
    store.save(run, as_baseline=args.set_baseline)

    print(f"run {run.run_id}  agent={run.agent}")
    print(f"  pass rate  {run.pass_rate:.1%}  ({sum(c.passed for c in run.cases)}/{len(run.cases)})")
    print(f"  mean score {run.mean_score:.3f}")
    for tag, rate in sorted(run.by_tag().items(), key=lambda kv: kv[1]):
        print(f"    {tag:<16} {rate:.1%}")
    for c in run.cases:
        if not c.passed:
            why = "; ".join(g.detail for g in c.grades if not g.passed and g.detail)
            print(f"  FAIL {c.case_id}: {why}")

    if args.set_baseline:
        print("\nbaseline set")
        return 0

    baseline = store.load_baseline()
    if baseline is None:
        print("\nno baseline stored; rerun with --set-baseline")
        return 0
    report = compare(baseline, run, args.tolerance)
    print()
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
