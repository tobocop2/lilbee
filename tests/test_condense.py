"""Tests for validating a model's history-condensation rewrite."""

from __future__ import annotations

from lilbee.retrieval.query.expansion import choose_retrieval_query

QUESTION = "and when was it written?"
CONTEXT = (
    "user: who kept the lighthouse journal at Split Rock\nassistant: It was kept by E. Larsen [1]."
)


def _chosen(reply: str, question: str = QUESTION, context: str = CONTEXT) -> str:
    return choose_retrieval_query(reply, question, context)


class TestAcceptedRewrites:
    def test_clean_rewrite_is_used(self):
        assert _chosen("when was the Split Rock journal written") == (
            "when was the Split Rock journal written"
        )

    def test_quoted_rewrite_is_unwrapped(self):
        assert _chosen('"when was the Split Rock journal written"') == (
            "when was the Split Rock journal written"
        )

    def test_query_after_a_preamble_line_is_used(self):
        reply = "Sure, here is the standalone query:\nwhen was the journal written"
        assert _chosen(reply) == "when was the journal written"

    def test_query_after_an_inline_preamble_is_used(self):
        reply = "Sure, here is the standalone query: when was the journal written"
        assert _chosen(reply) == "when was the journal written"

    def test_colon_from_the_conversation_is_kept(self):
        """A colon is only a preamble marker when the words before it are new.
        'Split Rock: the journal' names something the conversation used."""
        assert _chosen("Split Rock: the journal and when it was written") == (
            "Split Rock: the journal and when it was written"
        )

    def test_pronoun_only_question_anchors_on_the_conversation(self):
        """A question with no content words of its own ('what about that?')
        can only be anchored against the history it refers to."""
        assert _chosen("the Split Rock journal keeper", question="what about that?") == (
            "the Split Rock journal keeper"
        )

    def test_rewrite_equal_to_the_question_is_used(self):
        assert _chosen(QUESTION) == QUESTION

    def test_abbreviation_is_not_a_sentence_break(self):
        """A title or initialism ends in a period without ending a sentence.
        Rejecting those would drop exactly the proper-noun rewrites
        condensation exists to produce."""
        assert _chosen("when Dr. Larsen's journal was written") == (
            "when Dr. Larsen's journal was written"
        )
        assert _chosen("when the U.S. lighthouse journal was written") == (
            "when the U.S. lighthouse journal was written"
        )

    def test_single_cjk_sentence_is_used(self):
        assert _chosen("灯台の日誌", question="灯台の日誌", context="") == "灯台の日誌"


class TestRejectedRewrites:
    def test_lead_in_only_falls_back(self):
        assert _chosen("Sure, here is the standalone query:") == QUESTION

    def test_refusal_falls_back(self):
        assert _chosen("I cannot help with that request.") == QUESTION

    def test_reasoning_prose_falls_back(self):
        reply = (
            "The user is asking when the journal was written, so the standalone "
            "search query should mention the lighthouse journal at Split Rock."
        )
        assert _chosen(reply) == QUESTION

    def test_multiple_sentences_fall_back(self):
        assert _chosen("when was the journal written. It was kept by Larsen.") == QUESTION

    def test_two_questions_fall_back(self):
        """A question mark ends a sentence whatever precedes it. The word
        before the first terminator here is one character, which the period's
        abbreviation guard must not extend to."""
        assert _chosen("was it X? And when was the journal written?") == QUESTION

    def test_multiple_cjk_sentences_fall_back(self):
        """CJK sentences run on without a space after the terminator, so the
        break rule cannot require one."""
        question = "灯台の日誌"
        assert _chosen("灯台の日誌。次の質問", question=question, context="") == question

    def test_unrelated_query_falls_back(self):
        """A rewrite sharing no word with the question or the history is not a
        rewrite of anything the user asked."""
        assert _chosen("harbor tides in November") == QUESTION

    def test_empty_reply_falls_back(self):
        assert _chosen("") == QUESTION

    def test_whitespace_reply_falls_back(self):
        assert _chosen("   \n  \n") == QUESTION

    def test_overlong_rewrite_falls_back(self):
        reply = " ".join(["written"] * 40)
        assert _chosen(reply) == QUESTION

    def test_length_bound_scales_with_the_question(self):
        """The same rewrite is prose for a four-word question and a fair
        condensation of a long one."""
        reply = (
            "harbor survey findings for the northern shoreline erosion and tide "
            "gauge readings taken in the winter of 1902"
        )
        short = "what about the harbor?"
        long = "what did the harbor survey findings say about shoreline erosion and tide gauges"
        assert _chosen(reply, question=short) == short
        assert _chosen(reply, question=long) == reply
