# ragbot

A grounded RAG chatbot with citations, tool calling, and a regression gate.

Two things make a retrieval chatbot usable in production rather than impressive
in a demo: it cites what it used, and it refuses when the corpus can't support
an answer. Fluent, confident, uncited answers are the failure mode that gets a
support bot pulled from production.

`ragbot` does hybrid retrieval (dense + lexical), grounds every answer in
retrieved context with numbered citations, abstains below a similarity
threshold, and runs a bounded tool-calling loop. It is evaluated by
[agenteval](../agent-eval-harness), so prompt and model changes are gated by a
regression suite rather than by vibes.

Zero required dependencies. Runs offline with a deterministic embedder and an
extractive backend, so the full test suite needs no API key.

---

## Quick start

```bash
pip install -e ".[dev]"

ragbot ask --corpus corpus --question "How long do refunds take?"
ragbot chat --corpus corpus          # interactive, multi-turn
```

```python
from ragbot import RagChatbot

bot = RagChatbot()
bot.ingest_path("corpus")

answer = bot.chat("When is shipping free?")
print(answer.text)                 # "[1] Shipping is free on orders over 50 dollars..."
print(answer.citations[0].source)  # "shipping#1"
print(answer.grounded)             # True
```

Point it at a real model by swapping one object:

```python
from ragbot import RagChatbot, OpenAILLM, OpenAIEmbedder
bot = RagChatbot(llm=OpenAILLM(), embedder=OpenAIEmbedder())
```

---

## How it works

```
question
   ├─ embed ──────────────► dense cosine search   ┐
   └─ content tokens ─────► lexical overlap       ├─ blend (alpha) ─► top-k
                                                  ┘        │
                                          below min_score? ─┴─► abstain
                                                           │
                       numbered context + history ─► LLM ──┤
                                                           ├─ tool_call? ─► run tool ─┐
                                                           │        (max 3 iterations)┘
                                                           └─► grounded answer + citations
```

**Hybrid retrieval.** Dense embeddings miss exact identifiers — error codes,
SKUs, policy names — because they smooth them away. Lexical overlap catches
those but misses paraphrase. `alpha` blends the two; 1.0 is pure vector search,
0.0 pure keyword.

**Abstention.** Below `min_score` the bot returns no context and the prompt
instructs it to decline. Refusing is a feature, and the eval suite protects it:
loosening a prompt to sound more helpful is exactly what breaks it.

**Bounded tool loop.** `max_tool_iterations` caps the cycle. Without it, a model
that keeps requesting a failing tool loops until the budget is gone — the most
common way an agent hangs in production. Tool errors are returned to the model
as text so it can recover mid-turn instead of crashing.

---

## What the eval caught

The evaluation suite was not a formality — it found real bugs. First run:

```
pass rate  71.4%  (10/14)
    out-of-scope     0.0%
  FAIL oos-capital:     expected an abstention, got "[1] Invoices are available under Billing..."
  FAIL oos-weather:     expected an abstention, got "[1] Reset your password from the sign-in page..."
  FAIL oos-competitor:  expected an abstention, got "[1] Refunds are processed within 5 business days..."
```

Every out-of-scope question was answered from unrelated context. Two causes:

1. **Stopwords inflated lexical overlap.** "What is the capital of France?"
   shares *what / is / the / of* with every chunk in the corpus, so it scored
   0.26 — higher than a real question like "How do I reset my password?" at
   0.09. No threshold can separate those. Fixed by scoring only content-bearing
   tokens.
2. **Chunks were whole documents.** Every query retrieved something plausible.
   Fixed by chunking at 350 characters, which sharpened the score gap.

After both fixes, out-of-scope scores fell to 0.10–0.12 while real questions sat
at 0.18–0.44, so `min_score = 0.15` separates them cleanly.

```
pass rate  71.4% -> 85.7%
  FIXED  oos-capital   0.83 -> 1.00
  FIXED  oos-weather   0.83 -> 1.00
```

The gate also caught a **regression** the same change introduced
(`ship-standard`, 1.00 → 0.83) and correctly failed the build until it was
fixed. That is the whole point.

### Known limitations

Two golden cases still fail, and they are kept in the suite rather than deleted:

| Case | Why |
|---|---|
| `billing-dunning` | Needs synthesis across a chunk boundary. The extractive demo backend quotes; it cannot compose. Passes with a real LLM. |
| `oos-competitor` | "refund" and "policy" *are* corpus terms, so retrieval cannot reject the question. Refusal has to come from the generator reasoning about the word "competitor" — which a backend with no language model cannot do. |

A test suite that only contains passing tests is not measuring anything.

---

## Evaluation

```bash
pip install -e ../agent-eval-harness

ragbot eval --corpus corpus --goldens goldens/ragbot.jsonl --set-baseline
ragbot eval --corpus corpus --goldens goldens/ragbot.jsonl   # exit 1 on regression
```

Graders: the four generic ones from agenteval (no-error, contains, tool
trajectory, JSON), plus three specific to grounded retrieval:

- **`HasCitation`** — an assertion without a citation marker fails. Abstentions
  are exempt, since punishing them would train the system toward confident
  guessing.
- **`Abstains`** — out-of-corpus questions must be declined.
- **`NoToolLoop`** — flags a turn that hit the tool-iteration cap.

---

## Layout

```
src/ragbot/
  chunking.py      paragraph-aware splitting with overlap
  embeddings.py    HashingEmbedder (offline, deterministic) + OpenAI
  retriever.py     vector store, hybrid dense/lexical search, abstention
  llm.py           LLM protocol, ScriptedLLM, EchoContextLLM, OpenAI, Anthropic
  tools.py         registry, safe AST calculator, record lookup
  chat.py          retrieval, grounded generation, tool loop, memory
  eval_adapter.py  agenteval bridge + RAG-specific graders
  cli.py           ask / chat / eval
corpus/            sample knowledge base
goldens/           14 evaluation cases
tests/             58 tests, no network required
```

## Tests

```bash
pytest          # 58 passed
```

## License

MIT
