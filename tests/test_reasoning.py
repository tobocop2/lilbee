"""Tests for reasoning token filter: <think>...</think> tag detection."""

from lilbee.retrieval.reasoning import StreamToken, filter_reasoning, strip_reasoning

_NO_CAP = 1_000_000


def _collect(
    tokens: list[str],
    *,
    show: bool,
    cap_chars: int = _NO_CAP,
) -> list[StreamToken]:
    return list(filter_reasoning(iter(tokens), show=show, cap_chars=cap_chars))


class TestFilterReasoningShowFalse:
    def test_clean_text_passes_through(self):
        result = _collect(["Hello ", "world"], show=False)
        assert len(result) == 2
        assert all(not st.is_reasoning for st in result)
        assert "".join(st.content for st in result) == "Hello world"

    def test_strips_thinking_block(self):
        result = _collect(["<think>reasoning</think>answer"], show=False)
        content = "".join(st.content for st in result)
        assert "<think>" not in content
        assert "reasoning" not in content
        assert "answer" in content

    def test_strips_thinking_across_tokens(self):
        result = _collect(["<thi", "nk>deep thought</thi", "nk>final"], show=False)
        content = "".join(st.content for st in result)
        assert "deep thought" not in content
        assert "final" in content

    def test_content_before_and_after(self):
        result = _collect(["before<think>middle</think>after"], show=False)
        content = "".join(st.content for st in result)
        assert content == "beforeafter"

    def test_empty_thinking_block(self):
        result = _collect(["<think></think>answer"], show=False)
        content = "".join(st.content for st in result)
        assert content == "answer"

    def test_no_tokens(self):
        result = _collect([], show=False)
        assert result == []


class TestFilterReasoningShowTrue:
    def test_clean_text_not_reasoning(self):
        result = _collect(["Hello"], show=True)
        assert len(result) == 1
        assert result[0].content == "Hello"
        assert result[0].is_reasoning is False

    def test_thinking_yielded_as_reasoning(self):
        result = _collect(["<think>reasoning</think>answer"], show=True)
        reasoning = [st for st in result if st.is_reasoning]
        response = [st for st in result if not st.is_reasoning]
        assert len(reasoning) >= 1
        assert "reasoning" in "".join(st.content for st in reasoning)
        assert "answer" in "".join(st.content for st in response)

    def test_thinking_across_token_boundaries(self):
        tokens = ["<th", "ink>", "deep ", "thought", "</th", "ink>", "answer"]
        result = _collect(tokens, show=True)
        reasoning_text = "".join(st.content for st in result if st.is_reasoning)
        response_text = "".join(st.content for st in result if not st.is_reasoning)
        assert "deep thought" in reasoning_text
        assert "answer" in response_text

    def test_content_before_thinking(self):
        result = _collect(["before<think>thinking</think>after"], show=True)
        parts = [(st.content, st.is_reasoning) for st in result]
        before = [c for c, r in parts if not r and "before" in c]
        assert len(before) >= 1

    def test_empty_thinking_block(self):
        result = _collect(["<think></think>answer"], show=True)
        response = "".join(st.content for st in result if not st.is_reasoning)
        assert "answer" in response

    def test_multiple_thinking_blocks(self):
        result = _collect(["<think>first</think>mid<think>second</think>end"], show=True)
        reasoning = "".join(st.content for st in result if st.is_reasoning)
        response = "".join(st.content for st in result if not st.is_reasoning)
        assert "first" in reasoning
        assert "second" in reasoning
        assert "mid" in response
        assert "end" in response


class TestCouldBePartial:
    def test_partial_open_tag(self):
        result = _collect(["text<thi", "nk>reasoning</think>done"], show=False)
        content = "".join(st.content for st in result)
        assert "reasoning" not in content
        assert "text" in content
        assert "done" in content

    def test_partial_close_tag(self):
        result = _collect(["<think>thought</thi", "nk>done"], show=True)
        reasoning = "".join(st.content for st in result if st.is_reasoning)
        assert "thought" in reasoning

    def test_false_partial_not_tag(self):
        result = _collect(["text<", "b>not a tag"], show=False)
        content = "".join(st.content for st in result)
        assert "text" in content

    def test_unterminated_thinking_flushed_when_show(self):
        result = _collect(["<think>unterminated"], show=True)
        reasoning = [st for st in result if st.is_reasoning]
        assert len(reasoning) >= 1
        assert "unterminated" in "".join(st.content for st in reasoning)

    def test_unterminated_thinking_with_partial_close(self):
        result = _collect(["<think>deep thought</thi"], show=True)
        reasoning = "".join(st.content for st in result if st.is_reasoning)
        assert "deep thought" in reasoning

    def test_unterminated_thinking_stripped_when_hidden(self):
        result = _collect(["<think>unterminated"], show=False)
        content = "".join(st.content for st in result)
        assert content == ""

    def test_trailing_text_after_thinking(self):
        result = _collect(["<think>thought</think>trailing"], show=True)
        response = "".join(st.content for st in result if not st.is_reasoning)
        assert "trailing" in response

    def test_normal_text_ending_with_partial_tag(self):
        result = _collect(["hello<t"], show=False)
        content = "".join(st.content for st in result)
        assert "hello<t" in content


class TestReasoningCap:
    def test_cap_fires_on_long_reasoning(self):
        """Cap callback fires when reasoning exceeds cap_chars."""
        captured: list[str] = []
        long_think = "x" * 1500
        tokens = list(f"<think>{long_think}</think>answer")
        list(
            filter_reasoning(
                iter(tokens),
                show=True,
                cap_chars=1024,
                on_cap=captured.append,
            )
        )
        assert captured, "on_cap should fire when reasoning exceeds cap"
        assert "x" in captured[0]

    def test_cap_yields_no_response_after_fire(self):
        """When the cap fires, iteration stops; later tokens don't reach the consumer."""
        long_think = "x" * 1500
        tokens = list(f"<think>{long_think}</think>answer")
        result = list(
            filter_reasoning(
                iter(tokens),
                show=True,
                cap_chars=1024,
                on_cap=lambda _: None,
            )
        )
        response = "".join(st.content for st in result if not st.is_reasoning)
        assert "answer" not in response

    def test_no_cap_when_zero(self):
        """cap_chars=0 disables the cap entirely; no on_cap fires."""
        captured: list[str] = []
        long_think = "x" * 5000
        tokens = list(f"<think>{long_think}</think>answer")
        list(
            filter_reasoning(
                iter(tokens),
                show=False,
                cap_chars=0,
                on_cap=captured.append,
            )
        )
        assert captured == []

    def test_on_cap_optional(self):
        """Cap firing without an on_cap callback still terminates cleanly."""
        long_think = "x" * 1500
        tokens = list(f"<think>{long_think}</think>answer")
        result = list(filter_reasoning(iter(tokens), show=False, cap_chars=512))
        response = "".join(st.content for st in result if not st.is_reasoning)
        assert response == ""

    def test_on_progress_fires_during_reasoning(self):
        """on_progress receives running reasoning-chars counts as content arrives."""
        progress: list[int] = []
        # Ten tokens of 100 chars each: enough to cross the 256-char tick boundary.
        chunks = ["<think>"] + ["x" * 100 for _ in range(10)] + ["</think>", "answer"]
        list(
            filter_reasoning(
                iter(chunks),
                show=False,
                cap_chars=_NO_CAP,
                on_progress=progress.append,
            )
        )
        assert progress, "on_progress should fire as reasoning accumulates"
        assert progress == sorted(progress), "progress values should be monotonic"
        assert progress[-1] >= 1000

    def test_on_progress_optional(self):
        """Stream completes cleanly when on_progress is omitted."""
        result = _collect(["<think>x" * 200 + "</think>answer"], show=False)
        response = "".join(st.content for st in result if not st.is_reasoning)
        assert response == "answer"

    def test_captured_text_matches_partial_reasoning(self):
        """on_cap's payload is the reasoning content seen so far, not the cap value."""
        captured: list[str] = []
        partial = "thinking step one. " * 80
        tokens = [f"<think>{partial}"]
        list(
            filter_reasoning(
                iter(tokens),
                show=False,
                cap_chars=512,
                on_cap=captured.append,
            )
        )
        assert captured
        assert "thinking step one" in captured[0]

    def test_unclosed_think_terminates_show_false(self):
        """An unclosed <think> with show=False still hits the cap, no hang."""
        captured: list[str] = []
        tokens = ["<think>"] + ["x"] * 2000
        list(
            filter_reasoning(
                iter(tokens),
                show=False,
                cap_chars=512,
                on_cap=captured.append,
            )
        )
        assert captured

    def test_upstream_generator_closed_on_cap(self):
        """Cap firing closes the upstream iterator, releasing llama.cpp's chat lock."""
        closed = {"value": False}

        def runaway_tokens():
            try:
                yield "<think>"
                while True:
                    yield "x"
            except GeneratorExit:
                closed["value"] = True
                raise

        list(
            filter_reasoning(
                runaway_tokens(),
                show=True,
                cap_chars=512,
                on_cap=lambda _: None,
            )
        )
        assert closed["value"], "filter_reasoning must close upstream iterator"

    def test_close_called_on_stream_wrapper_early_exit(self):
        """A stream wrapper exposing close() has it called when the cap fires."""

        class FakeStream:
            def __init__(self) -> None:
                self.tokens = iter(["<think>"] + ["x"] * 2000)
                self.closed = False

            def __iter__(self):
                return self

            def __next__(self):
                return next(self.tokens)

            def close(self) -> None:
                self.closed = True

        stream = FakeStream()
        list(
            filter_reasoning(
                stream,
                show=True,
                cap_chars=512,
                on_cap=lambda _: None,
            )
        )
        assert stream.closed


class TestStripReasoning:
    def test_strips_think_block(self):
        assert strip_reasoning("<think>internal</think>answer") == "answer"

    def test_no_think_block(self):
        assert strip_reasoning("plain text") == "plain text"

    def test_multiple_blocks(self):
        text = "<think>a</think>one<think>b</think>two"
        assert strip_reasoning(text) == "onetwo"

    def test_multiline_think(self):
        text = "<think>\nline1\nline2\n</think>\nresult"
        assert strip_reasoning(text) == "result"

    def test_empty_string(self):
        assert strip_reasoning("") == ""

    def test_trailing_whitespace_after_tag_stripped(self):
        assert strip_reasoning("<think>x</think>  answer") == "answer"

    def test_unclosed_think_tag_stripped(self):
        assert strip_reasoning("answer<think>truncated reasoning") == "answer"

    def test_only_unclosed_think_tag(self):
        assert strip_reasoning("<think>all reasoning no answer") == ""
