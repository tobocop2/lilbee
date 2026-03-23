"""Tests for reasoning token filter — <think>...</think> tag detection."""

from lilbee.reasoning import StreamToken, filter_reasoning


def _collect(tokens: list[str], *, show: bool) -> list[StreamToken]:
    return list(filter_reasoning(iter(tokens), show=show))


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
        """Token ending with '<thi' should wait for more data."""
        result = _collect(["text<thi", "nk>reasoning</think>done"], show=False)
        content = "".join(st.content for st in result)
        assert "reasoning" not in content
        assert "text" in content
        assert "done" in content

    def test_partial_close_tag(self):
        """Token ending with '</thi' inside thinking should wait."""
        result = _collect(["<think>thought</thi", "nk>done"], show=True)
        reasoning = "".join(st.content for st in result if st.is_reasoning)
        assert "thought" in reasoning

    def test_false_partial_not_tag(self):
        """'<' at end that doesn't match tag prefix should not stall."""
        result = _collect(["text<", "b>not a tag"], show=False)
        content = "".join(st.content for st in result)
        assert "text" in content

    def test_unterminated_thinking_flushed_when_show(self):
        """Thinking block never closed — buffer flushed as reasoning at end."""
        result = _collect(["<think>unterminated"], show=True)
        reasoning = [st for st in result if st.is_reasoning]
        assert len(reasoning) >= 1
        assert "unterminated" in "".join(st.content for st in reasoning)

    def test_unterminated_thinking_with_partial_close(self):
        """Thinking with partial close tag at end — flushed as reasoning."""
        result = _collect(["<think>deep thought</thi"], show=True)
        reasoning = "".join(st.content for st in result if st.is_reasoning)
        assert "deep thought" in reasoning

    def test_unterminated_thinking_stripped_when_hidden(self):
        """Thinking block never closed — buffer discarded when show=False."""
        result = _collect(["<think>unterminated"], show=False)
        content = "".join(st.content for st in result)
        assert content == ""

    def test_trailing_text_after_thinking(self):
        """Normal text at end of stream flushed correctly."""
        result = _collect(["<think>thought</think>trailing"], show=True)
        response = "".join(st.content for st in result if not st.is_reasoning)
        assert "trailing" in response

    def test_normal_text_ending_with_partial_tag(self):
        """Buffer has normal text ending with '<t' — flushed as normal at end."""
        result = _collect(["hello<t"], show=False)
        content = "".join(st.content for st in result)
        assert "hello<t" in content
