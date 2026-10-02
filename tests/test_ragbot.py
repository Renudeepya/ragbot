import json

import pytest

from ragbot import (
    Chunk, HashingEmbedder, HybridRetriever, Message, RagChatbot, ScriptedLLM,
    Tool, ToolError, ToolRegistry, VectorStore, calculator, chunk_document,
    cosine, lexical_overlap, make_calculator_tool, make_lookup_tool,
    parse_tool_calls,
)
from ragbot.llm import LLMResponse

CORPUS = "corpus"


# ------------------------------------------------------------------ chunking
class TestChunking:
    def test_splits_on_paragraphs(self):
        text = "First para.\n\nSecond para.\n\nThird para."
        chunks = chunk_document("d", text, chunk_size=20, overlap=0)
        assert len(chunks) == 3
        assert chunks[0].text == "First para."

    def test_packs_small_paragraphs_together(self):
        text = "aaa.\n\nbbb.\n\nccc."
        chunks = chunk_document("d", text, chunk_size=500)
        assert len(chunks) == 1

    def test_hard_splits_oversized_paragraph(self):
        chunks = chunk_document("d", "x" * 2000, chunk_size=500, overlap=50)
        assert len(chunks) > 1
        assert all(len(c.text) <= 500 for c in chunks)

    def test_ordinals_and_ids_are_sequential(self):
        chunks = chunk_document("doc", "a.\n\nb.\n\nc.", chunk_size=5, overlap=0)
        assert [c.ordinal for c in chunks] == [0, 1, 2]
        assert [c.id for c in chunks] == ["doc::0", "doc::1", "doc::2"]

    def test_empty_document_yields_nothing(self):
        assert chunk_document("d", "   \n\n  ") == []

    def test_rejects_invalid_params(self):
        with pytest.raises(ValueError):
            chunk_document("d", "x", chunk_size=0)
        with pytest.raises(ValueError):
            chunk_document("d", "x", chunk_size=10, overlap=10)

    def test_citation_uses_title_metadata(self):
        c = chunk_document("d", "text", metadata={"title": "Refunds"})[0]
        assert c.citation == "Refunds#0"


# ---------------------------------------------------------------- embeddings
class TestEmbeddings:
    def test_deterministic(self):
        e = HashingEmbedder(dim=64)
        assert e.embed(["hello world"]) == e.embed(["hello world"])

    def test_unit_normalized(self):
        v = HashingEmbedder(dim=64).embed(["some text here"])[0]
        assert sum(x * x for x in v) == pytest.approx(1.0, abs=1e-6)

    def test_similar_text_scores_higher_than_unrelated(self):
        e = HashingEmbedder(dim=512)
        a, b, c = e.embed([
            "refunds are processed within 5 business days",
            "refunds are processed within five business days",
            "two-factor authentication uses hardware keys",
        ])
        assert cosine(a, b) > cosine(a, c)

    def test_empty_text_is_zero_vector(self):
        assert all(x == 0.0 for x in HashingEmbedder(dim=32).embed([""])[0])

    def test_cosine_rejects_mismatched_dims(self):
        with pytest.raises(ValueError):
            cosine([1.0, 0.0], [1.0, 0.0, 0.0])

    def test_rejects_bad_dim(self):
        with pytest.raises(ValueError):
            HashingEmbedder(dim=0)


# ------------------------------------------------------------------ retrieval
class TestRetrieval:
    def make(self):
        r = HybridRetriever(HashingEmbedder(dim=512), k=3, min_score=0.0)
        r.index(chunk_document("refunds", "Refunds are processed within 5 business days."))
        r.index(chunk_document("shipping", "Standard shipping takes 3 to 5 business days."))
        r.index(chunk_document("auth", "Reset your password using the Forgot password link."))
        return r

    def test_retrieves_relevant_chunk_first(self):
        hits = self.make().retrieve("how do I reset my password")
        assert hits[0].chunk.doc_id == "auth"

    def test_respects_k(self):
        assert len(self.make().retrieve("business days", k=2)) == 2

    def test_empty_store_returns_nothing(self):
        assert HybridRetriever(HashingEmbedder(dim=32)).retrieve("anything") == []

    def test_min_score_filters_weak_matches(self):
        r = HybridRetriever(HashingEmbedder(dim=512), min_score=0.99)
        r.index(chunk_document("d", "completely unrelated content about gardening"))
        assert r.retrieve("quantum chromodynamics") == []

    def test_lexical_overlap_bounds(self):
        assert lexical_overlap("alpha beta", "alpha beta gamma") == 1.0
        assert lexical_overlap("alpha beta", "gamma delta") == 0.0
        assert lexical_overlap("", "anything") == 0.0

    def test_alpha_zero_is_pure_lexical(self):
        r = HybridRetriever(HashingEmbedder(dim=512), alpha=0.0, min_score=0.0)
        r.index(chunk_document("a", "zebra quokka"))
        r.index(chunk_document("b", "nothing matching here"))
        assert r.retrieve("zebra quokka")[0].chunk.doc_id == "a"

    def test_rejects_bad_alpha(self):
        with pytest.raises(ValueError):
            HybridRetriever(HashingEmbedder(dim=32), alpha=1.5)

    def test_reingesting_replaces_chunks(self):
        r = HybridRetriever(HashingEmbedder(dim=128))
        r.index(chunk_document("d", "original text"))
        r.index(chunk_document("d", "original text"))
        assert len(r.store) == 1


class TestVectorStore:
    def test_round_trip(self, tmp_path):
        e = HashingEmbedder(dim=64)
        chunks = chunk_document("d", "hello world")
        store = VectorStore(64)
        store.add(chunks, e.embed([c.text for c in chunks]))
        path = store.save(tmp_path / "s.json")

        loaded = VectorStore.load(path)
        assert len(loaded) == len(store)
        assert loaded.chunks[0].id == chunks[0].id

    def test_rejects_wrong_dim(self):
        store = VectorStore(4)
        with pytest.raises(ValueError):
            store.add(chunk_document("d", "x"), [[0.0, 0.0]])

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError):
            VectorStore(2).add(chunk_document("d", "x"), [])


# ---------------------------------------------------------------------- tools
class TestTools:
    def test_calculator(self):
        assert calculator("3 * (4 + 5)") == "27"
        assert calculator("-2 ** 3") == "-8"

    def test_calculator_blocks_code_execution(self):
        with pytest.raises(ToolError):
            calculator("__import__('os').system('echo pwned')")

    def test_calculator_reports_division_by_zero(self):
        with pytest.raises(ToolError):
            calculator("1/0")

    def test_registry_invoke(self):
        reg = ToolRegistry([make_calculator_tool()])
        assert reg.invoke("calculator", {"expression": "2+2"}) == "4"

    def test_unknown_tool_returns_error_not_raise(self):
        assert "unknown tool" in ToolRegistry().invoke("nope", {})

    def test_bad_arguments_return_error(self):
        reg = ToolRegistry([make_calculator_tool()])
        assert "bad arguments" in reg.invoke("calculator", {"wrong": 1})

    def test_tool_error_surfaces_to_model(self):
        reg = ToolRegistry([make_lookup_tool({"1": {"status": "shipped"}})])
        assert "unknown order id" in reg.invoke("lookup_order", {"order_id": "999"})

    def test_lookup_returns_json(self):
        reg = ToolRegistry([make_lookup_tool({"1": {"status": "shipped"}})])
        assert json.loads(reg.invoke("lookup_order", {"order_id": "1"}))["status"] == "shipped"

    def test_duplicate_registration_rejected(self):
        reg = ToolRegistry([make_calculator_tool()])
        with pytest.raises(ValueError):
            reg.register(make_calculator_tool())


# ------------------------------------------------------------------------ llm
class TestToolCallParsing:
    def test_parses_block(self):
        text, calls = parse_tool_calls(
            'Let me check. <tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>'
        )
        assert text == "Let me check."
        assert calls == [{"name": "calculator", "arguments": {"expression": "1+1"}}]

    def test_malformed_json_left_as_prose(self):
        text, calls = parse_tool_calls("<tool_call>{not json}</tool_call>")
        assert calls == [] and "tool_call" in text

    def test_no_block(self):
        assert parse_tool_calls("plain answer") == ("plain answer", [])

    def test_multiple_blocks(self):
        _, calls = parse_tool_calls(
            '<tool_call>{"name": "a"}</tool_call><tool_call>{"name": "b"}</tool_call>'
        )
        assert [c["name"] for c in calls] == ["a", "b"]


class TestScriptedLLM:
    def test_replays_in_order(self):
        llm = ScriptedLLM(["first", "second"])
        assert llm.complete([]).text == "first"
        assert llm.complete([]).text == "second"

    def test_falls_back_to_default(self):
        assert "don't have enough information" in ScriptedLLM([]).complete([]).text

    def test_records_messages(self):
        llm = ScriptedLLM(["x"])
        llm.complete([Message("user", "hi")])
        assert llm.calls[0][0].content == "hi"


# -------------------------------------------------------------------- chatbot
class TestChatbot:
    def bot(self, responses=None, tools=None):
        return RagChatbot(
            llm=ScriptedLLM(responses or []),
            embedder=HashingEmbedder(dim=512),
            tools=tools,
        )

    def test_ingests_corpus(self):
        b = self.bot()
        assert b.ingest_path(CORPUS) > 0

    def test_ingest_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            self.bot().ingest_path(CORPUS, pattern="*.nope")

    def test_retrieved_context_reaches_the_model(self):
        b = self.bot(["Refunds take 5 business days [1]."])
        b.ingest_path(CORPUS)
        b.chat("How long do refunds take?")
        system = [m.content for m in b.llm.calls[0] if m.role == "system"]
        assert any("CONTEXT" in s for s in system)
        assert any("business days" in s for s in system)

    def test_answer_carries_citations(self):
        b = self.bot(["Refunds take 5 business days [1]."])
        b.ingest_path(CORPUS)
        ans = b.chat("How long do refunds take?")
        assert ans.citations and ans.citations[0].marker == "[1]"
        assert ans.cited_markers == ["1"]
        assert ans.grounded

    def test_empty_index_is_not_grounded(self):
        ans = self.bot(["whatever"]).chat("anything")
        assert not ans.grounded and ans.citations == []

    def test_context_says_none_when_nothing_retrieved(self):
        b = self.bot(["x"])
        b.chat("anything")
        assert any("(none)" in m.content for m in b.llm.calls[0] if m.role == "system")

    def test_multi_turn_memory(self):
        b = self.bot(["Five business days [1].", "Yes, to the original method [1]."])
        b.ingest_path(CORPUS)
        b.chat("How long do refunds take?")
        b.chat("Does it go back to my card?")
        roles = [(m.role, m.content) for m in b.llm.calls[1]]
        assert ("user", "How long do refunds take?") in roles

    def test_reset_clears_history(self):
        b = self.bot(["a", "b"])
        b.chat("one")
        b.reset()
        b.chat("two")
        assert len(b.llm.calls[1]) == len(b.llm.calls[0])

    def test_history_is_capped(self):
        b = RagChatbot(llm=ScriptedLLM(["x"] * 20), embedder=HashingEmbedder(dim=64), max_history=2)
        for i in range(6):
            b.chat(f"q{i}")
        convo = [m for m in b.llm.calls[-1] if m.role in ("user", "assistant")]
        assert len(convo) <= 3  # 2 history + current question

    def test_latency_recorded(self):
        assert self.bot(["x"]).chat("q").latency_ms >= 0


class TestToolLoop:
    def bot(self, responses):
        return RagChatbot(
            llm=ScriptedLLM(responses),
            embedder=HashingEmbedder(dim=128),
            tools=ToolRegistry([make_calculator_tool()]),
        )

    def test_executes_tool_and_continues(self):
        b = self.bot([
            '<tool_call>{"name": "calculator", "arguments": {"expression": "6*7"}}</tool_call>',
            "The answer is 42.",
        ])
        ans = b.chat("what is 6 times 7")
        assert ans.text == "The answer is 42."
        assert ans.tool_calls[0]["name"] == "calculator"
        assert ans.tool_calls[0]["result"] == "42"

    def test_tool_result_is_fed_back(self):
        b = self.bot([
            '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>',
            "It is 4.",
        ])
        b.chat("2+2?")
        assert any(m.role == "tool" and m.content == "4" for m in b.llm.calls[1])

    def test_tool_error_reaches_model_without_crashing(self):
        b = self.bot([
            '<tool_call>{"name": "calculator", "arguments": {"expression": "1/0"}}</tool_call>',
            "Sorry, that is undefined.",
        ])
        ans = b.chat("1/0?")
        assert "error" in ans.tool_calls[0]["result"]
        assert ans.text == "Sorry, that is undefined."

    def test_iteration_cap_stops_runaway_loop(self):
        looping = '<tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>'
        b = RagChatbot(
            llm=ScriptedLLM([looping] * 10),
            embedder=HashingEmbedder(dim=64),
            tools=ToolRegistry([make_calculator_tool()]),
            max_tool_iterations=2,
        )
        ans = b.chat("loop forever")
        assert any("iteration limit reached" in c["result"] for c in ans.tool_calls)

    def test_no_tools_means_no_schemas_passed(self):
        b = RagChatbot(llm=ScriptedLLM(["plain"]), embedder=HashingEmbedder(dim=64))
        assert b.chat("hi").tool_calls == []


class TestGroundedBehaviour:
    """The abstention path — the defense against confident hallucination."""

    def test_echo_backend_abstains_without_context(self):
        from ragbot import EchoContextLLM

        bot = RagChatbot(llm=EchoContextLLM(), embedder=HashingEmbedder(dim=256))
        assert "don't have enough information" in bot.chat("capital of France?").text

    def test_echo_backend_answers_from_corpus(self):
        from ragbot import EchoContextLLM

        bot = RagChatbot(llm=EchoContextLLM(), embedder=HashingEmbedder(dim=512))
        bot.ingest_path(CORPUS)
        assert "business days" in bot.chat("How long do refunds take?").text.lower()

    def test_answer_serializes(self):
        b = RagChatbot(llm=ScriptedLLM(["answer [1]"]), embedder=HashingEmbedder(dim=256))
        b.ingest_path(CORPUS)
        d = b.chat("refunds?").to_dict()
        assert {"text", "citations", "tool_calls", "grounded"} <= d.keys()
