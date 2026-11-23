"""Tests for document title and source-metadata derivation at ingest."""

from contextlib import contextmanager

import pytest
from xberg import Metadata

from lilbee.core.web_names import WEB_PREFIX
from lilbee.crawler import url_to_filename
from lilbee.data.title import derive_title, source_meta_from_extraction
from lilbee.providers.engine_params import INPUT_HEADROOM_TOKENS


class TestDeriveTitle:
    def test_metadata_title_wins(self):
        assert derive_title("q3_report.pdf", "Quarterly Revenue Report") == (
            "Quarterly Revenue Report"
        )

    def test_metadata_title_is_stripped(self):
        assert derive_title("a.pdf", "  Padded Title \n") == "Padded Title"

    def test_blank_metadata_falls_back_to_stem(self):
        assert derive_title("survey_214.pdf", "   ") == "survey 214"

    def test_none_metadata_falls_back_to_stem(self):
        assert derive_title("annual-wildlife-survey.pdf") == "annual wildlife survey"

    def test_stem_keeps_meaningful_dots(self):
        assert derive_title("spec-v3.0.pdf") == "spec v3.0"

    def test_subdirectory_source_uses_basename_stem(self):
        assert derive_title("filings/2024/notice_of_appeal.pdf") == "notice of appeal"

    def test_mixed_separators_collapse(self):
        assert derive_title("a_b-c  d.txt") == "a b c d"

    def test_crawled_page_source_drops_the_address_hash(self):
        assert derive_title("_web/en.wikipedia.org/wiki/Cannabis~25c17501722158ba.md") == "Cannabis"
        assert derive_title("_web/docs.python.org/3/tutorial/index~a17e5b24c217aa80.md") == "index"

    def test_a_crawled_page_named_for_an_account_keeps_its_at_sign(self):
        url = "https://forum.example.com/@forum.growweedeasy.com/"
        assert derive_title(WEB_PREFIX + url_to_filename(url)) == "@forum.growweedeasy"

    def test_a_local_note_keeps_a_tilde_in_its_stem(self):
        assert derive_title("notes/plan~draft.md") == "plan~draft"


class TestSourceMetaFromExtraction:
    def test_full_metadata(self):
        meta = source_meta_from_extraction(
            Metadata(title="The Title", authors=["Ada", "Grace"], created_at="2021-05-01"),
            "x.pdf",
        )
        assert meta.title == "The Title"
        assert meta.authors == "Ada, Grace"
        assert meta.created_at == "2021-05-01"

    def test_empty_metadata_derives_stem_title(self):
        meta = source_meta_from_extraction(Metadata(), "field_notes.pdf")
        assert meta.title == "field notes"
        assert meta.authors == ""
        assert meta.created_at == ""

    def test_falsy_authors_are_dropped(self):
        meta = source_meta_from_extraction(Metadata(authors=["Ada", "", None]), "x.pdf")
        assert meta.authors == "Ada"

    def test_none_values_tolerated(self):
        meta = source_meta_from_extraction(
            Metadata(title=None, authors=None, created_at=None), "notes.md"
        )
        assert meta.title == "notes"
        assert meta.authors == ""
        assert meta.created_at == ""

    def test_string_authors_is_one_author_not_split_into_characters(self):
        # A raw PDF /Author field often arrives as a plain string; it must not
        # be iterated into "J, o, h, n".
        meta = source_meta_from_extraction(Metadata(authors="John Doe"), "x.pdf")
        assert meta.authors == "John Doe"

    def test_non_string_author_entries_are_coerced_not_raised(self):
        meta = source_meta_from_extraction(Metadata(authors=["Ada", 42]), "x.pdf")
        assert meta.authors == "Ada, 42"

    def test_non_string_title_falls_back_to_stem(self):
        # A bytes/number title in malformed metadata must not raise; fall back.
        meta = source_meta_from_extraction(Metadata(title=123), "annual_report.pdf")
        assert meta.title == "annual report"


class TestJunkStems:
    """Stems with no searchable words stay untitled so the title arm indexes no noise."""

    def test_counter_and_numeric_stems_yield_no_title(self):
        for name in [
            "IMG_1234.jpg",
            "DSC0001.png",
            "scan_001.pdf",
            "Screenshot 2024-01-02.png",
            "2024-03-15.md",
            "doc42.pdf",
            "a1b2c3d4e5f6a7b8.bin",
            "x.md",
        ]:
            assert derive_title(name) == "", name

    def test_a_crawled_page_named_by_a_bare_number_takes_it_as_title(self):
        assert derive_title("_web/forum.example.com/thread/12345~25c17501722158ba.md") == "12345"
        assert derive_title("_web/forum.example.com/thread/7~25c17501722158ba.md") == "7"

    def test_a_crawled_page_named_by_a_hex_string_takes_it_as_title(self):
        assert derive_title("_web/example.com/c/a1b2c3d4e5f6~25c17501722158ba.md") == "a1b2c3d4e5f6"
        assert derive_title("_web/example.com/c/DEADBEEF~25c17501722158ba.md") == "DEADBEEF"

    def test_other_junk_stems_of_a_crawled_page_stay_untitled(self):
        for name in [
            "IMG_1234",
            "scan_001",
            "2024-03-15",
            "2024 03 15",
            "doc42",
            "x",
            "ab",
            "a1b2c3d4-e5f6-a7b8",
            "1.5",
        ]:
            assert derive_title(f"_web/example.com/p/{name}~25c17501722158ba.md") == "", name

    def test_a_local_file_named_by_a_number_stays_untitled(self):
        assert derive_title("filings/12345.pdf") == ""

    def test_extracted_title_beats_a_numeric_page_name(self):
        name = "_web/forum.example.com/thread/12345~25c17501722158ba.md"
        assert derive_title(name, "Help with setup") == "Help with setup"

    def test_real_stems_survive(self):
        assert derive_title("project_falcon_notes.pdf") == "project falcon notes"
        assert derive_title("survey_214.pdf") == "survey 214"

    def test_extracted_title_bypasses_the_junk_guard(self):
        assert derive_title("IMG_1234.jpg", "Sunset over the harbor") == "Sunset over the harbor"


@contextmanager
def _counting_services(count_tokens, window: int | None = None):
    """Services whose embedder counts tokens with *count_tokens* and has the token *window*."""
    from unittest.mock import MagicMock

    from lilbee.app.services import set_services

    svc = MagicMock()
    svc.provider.count_tokens.side_effect = count_tokens
    svc.provider.embed_token_cap.return_value = window
    set_services(svc)
    try:
        yield svc
    finally:
        set_services(None)


def _no_tokenizer(text: str) -> int:
    raise NotImplementedError


def _one_token_a_character(text: str) -> int:
    return len(text)


_NOMIC_REF = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
_NOMIC_PREFIX = "search_document: "
_FULL_CHUNK_ESTIMATED = "w" * (512 * 3)
_FULL_CHUNK_COUNTED = "w" * 512
_INPUT_LIMIT = 640


class TestEmbedInputsKeepTheChunk:
    """A long title or sentence takes only the headroom, so the limit never cuts the chunk."""

    @pytest.fixture(autouse=True)
    def _titles_on_nomic(self, monkeypatch):
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "embed_titles", True)
        monkeypatch.setattr(cfg, "embedding_model", _NOMIC_REF)

    def _inputs(self, count_tokens, texts, title, sentences=None):
        from lilbee.data.extract.document import _embed_inputs

        with _counting_services(count_tokens):
            return _embed_inputs(texts, title, sentences)

    @pytest.mark.parametrize("title_chars", [368, 1000])
    def test_a_long_english_title_leaves_the_estimated_chunk_whole(self, title_chars):
        from lilbee.data.extract.backends.tokenizer import _estimate_tokens

        title = ("Annual report of the river authority " * 40)[:title_chars]
        [sent] = self._inputs(_no_tokenizer, [_FULL_CHUNK_ESTIMATED], title)
        assert sent.endswith(f"\n{_FULL_CHUNK_ESTIMATED}")
        assert sent.startswith("Annual report")
        assert _estimate_tokens(_NOMIC_PREFIX + sent) <= _INPUT_LIMIT

    def test_a_long_cjk_title_leaves_the_counted_chunk_whole(self):
        title = "河川管理局年次報告書" * 40
        [sent] = self._inputs(_one_token_a_character, [_FULL_CHUNK_COUNTED], title)
        assert sent.endswith(f"\n{_FULL_CHUNK_COUNTED}")
        assert sent.startswith("河川管理局")
        assert len(_NOMIC_PREFIX + sent) <= _INPUT_LIMIT

    def test_title_and_sentence_share_the_headroom_title_first(self):
        title = "T" * 100
        sentence = "S" * 100
        [sent] = self._inputs(_one_token_a_character, [_FULL_CHUNK_COUNTED], title, [sentence])
        assert sent.startswith(f"{title}\nS")
        assert sent.endswith(f"\n{_FULL_CHUNK_COUNTED}")
        assert len(_NOMIC_PREFIX + sent) <= _INPUT_LIMIT

    def test_a_sentence_alone_is_bounded_too(self, monkeypatch):
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "embed_titles", False)
        [sent] = self._inputs(_one_token_a_character, [_FULL_CHUNK_COUNTED], "ignored", ["S" * 300])
        assert sent.endswith(f"S\n{_FULL_CHUNK_COUNTED}")
        assert len(_NOMIC_PREFIX + sent) <= _INPUT_LIMIT

    def test_a_short_title_and_sentence_stay_whole(self):
        [sent] = self._inputs(_one_token_a_character, ["chunk"], "Annual Report", ["It covers Q3."])
        assert sent == "Annual Report\nIt covers Q3.\nchunk"

    def test_one_title_is_counted_once_for_all_chunks(self):
        from lilbee.data.extract.document import _embed_inputs

        with _counting_services(_one_token_a_character) as svc:
            inputs = _embed_inputs(["a", "b", "c"], "Annual Report")
        assert inputs == ["Annual Report\na", "Annual Report\nb", "Annual Report\nc"]
        counted = [call.args[0] for call in svc.provider.count_tokens.call_args_list]
        assert counted.count("Annual Report") == 1

    def test_the_head_budget_is_the_headroom_less_the_prefix_and_newline(self):
        from lilbee.data.extract.document import _head_budget
        from lilbee.providers.engine_params import INPUT_HEADROOM_TOKENS

        with _counting_services(_one_token_a_character):
            assert _head_budget().room == INPUT_HEADROOM_TOKENS - len(_NOMIC_PREFIX) - 1


_LOCAL_EMBEDDER = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q8_0.gguf"
_REMOTE_EMBEDDER = "openai/text-embedding-3-small"
_LONG_TITLE = "T" * 1000


class TestTheEmbedInputFollowsTheWindow:
    """One embed input is the prefix, the head and one whole chunk; the head takes only the
    room the embedder's window leaves after the chunk (one test per acceptance row)."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        from lilbee.core.config import cfg
        from lilbee.data.extract import document
        from lilbee.retrieval.embedding_profiles import EmbeddingProfile

        monkeypatch.setattr(cfg, "embed_titles", True)
        monkeypatch.setattr(cfg, "chunk_size", 512)
        profile = EmbeddingProfile(doc_prefix=_NOMIC_PREFIX)
        monkeypatch.setattr(document, "resolve_embedding_profile", lambda _ref: profile)
        monkeypatch.setattr(
            "lilbee.retrieval.embedding_profiles.resolve_embedding_profile", lambda _ref: profile
        )
        self.cfg = cfg
        self.monkeypatch = monkeypatch

    def _sent(self, window, *, remote=False):
        """(chunk, the full input with its prefix) for one longest chunk under a long title."""
        from lilbee.data.extract.chunk import token_budget
        from lilbee.data.extract.document import _embed_inputs

        model = _REMOTE_EMBEDDER if remote else _LOCAL_EMBEDDER
        self.monkeypatch.setattr(self.cfg, "embedding_model", model)
        count = _no_tokenizer if remote else _one_token_a_character
        with _counting_services(count, window):
            tokens = token_budget()
            chunk = "c" * (tokens * 3 if remote else tokens)
            [sent] = _embed_inputs([chunk], title=_LONG_TITLE)
        return chunk, _NOMIC_PREFIX + sent

    def test_a_small_window_that_holds_a_chunk_cuts_only_the_title(self):
        chunk, sent = self._sent(600)
        assert len(chunk) == 512
        assert sent.endswith("\n" + chunk)
        assert len(sent) == 600

    def test_a_window_below_one_chunk_refuses_with_the_working_size(self):
        from lilbee.data.extract.chunk import EmbedWindowError

        with pytest.raises(EmbedWindowError) as raised:
            self._sent(400)
        assert str(raised.value) == (
            f"{_LOCAL_EMBEDDER}: The embedding model accepts 400 tokens per input; "
            "chunk_size 512 does not fit. Set chunk_size to 272 or lower."
        )

    def test_a_window_with_room_gives_the_title_the_full_headroom(self):
        chunk, sent = self._sent(640)
        assert len(chunk) == 512
        assert sent.endswith("\n" + chunk)
        assert len(sent) == 640

    def test_an_unknown_window_keeps_the_headroom_budget(self):
        chunk, sent = self._sent(None)
        assert len(chunk) == 512
        assert sent.endswith("\n" + chunk)
        assert len(sent) == 512 + INPUT_HEADROOM_TOKENS

    def test_a_remote_embedder_fills_the_character_guard(self):
        from lilbee.retrieval.embedder import embed_char_budget

        chunk, sent = self._sent(None, remote=True)
        assert len(chunk) == 512 * 3
        assert sent.endswith("\n" + chunk)
        assert len(sent) == embed_char_budget(self.cfg)


class TestEmbedTitles:
    """Opt-in title prefixing of embedding inputs; stored chunk text unchanged."""

    @pytest.fixture(autouse=True)
    def _counted(self):
        with _counting_services(_one_token_a_character):
            yield

    def test_off_by_default_passes_texts_through(self):
        from lilbee.core.config import cfg
        from lilbee.data.extract.document import _embed_inputs

        assert cfg.embed_titles is False
        assert _embed_inputs(["chunk text"], "Annual Report") == ["chunk text"]

    def test_on_prefixes_the_title(self):
        from lilbee.core.config import cfg
        from lilbee.data.extract.document import _embed_inputs

        cfg.embed_titles = True
        try:
            assert _embed_inputs(["chunk text"], "Annual Report") == ["Annual Report\nchunk text"]
            assert _embed_inputs(["chunk text"], "") == ["chunk text"]
        finally:
            cfg.embed_titles = False

    def test_scoped_title_reaches_the_ocr_embed_path(self):
        from lilbee.core.config import cfg
        from lilbee.data.extract.document import _embed_inputs, _title_scope

        cfg.embed_titles = True
        try:
            with _title_scope("Scan Batch Q3"):
                assert _embed_inputs(["page text"]) == ["Scan Batch Q3\npage text"]
            assert _embed_inputs(["page text"]) == ["page text"]
        finally:
            cfg.embed_titles = False


class TestContextualEnrichment:
    """Opt-in per-chunk situating sentence, embedding input only."""

    def _services(self, reply="This chunk covers the budget section of the annual report."):
        from unittest.mock import MagicMock

        svc = MagicMock()
        svc.provider.chat.return_value = MagicMock(text=reply)
        return svc

    def test_off_by_default_is_passthrough(self):
        from lilbee.data.extract.document import _enrich_texts

        assert _enrich_texts(["chunk"], "doc head", "a.pdf") == [""]

    def test_on_writes_one_sentence_for_each_chunk(self):
        from unittest.mock import patch

        from lilbee.core.config import cfg
        from lilbee.data.extract.document import _enrich_texts

        svc = self._services()
        cfg.contextual_enrichment = True
        try:
            with patch("lilbee.data.extract.document.get_services", return_value=svc):
                out = _enrich_texts(["chunk text"], "doc head", "a.pdf")
        finally:
            cfg.contextual_enrichment = False
        assert out == ["This chunk covers the budget section of the annual report."]

    def test_turns_thinking_off(self):
        """One situating sentence on a small cap: a thinking model spends the
        cap deliberating and every chunk embeds bare."""
        from unittest.mock import patch

        from lilbee.core.config import cfg
        from lilbee.data.extract.document import _enrich_texts

        svc = self._services()
        cfg.contextual_enrichment = True
        try:
            with patch("lilbee.data.extract.document.get_services", return_value=svc):
                _enrich_texts(["chunk text"], "doc head", "a.pdf")
        finally:
            cfg.contextual_enrichment = False
        assert svc.provider.chat.call_args.kwargs["options"]["think"] is False

    def test_failure_keeps_the_bare_chunk(self):
        from unittest.mock import patch

        from lilbee.core.config import cfg
        from lilbee.data.extract.document import _enrich_texts

        svc = self._services()
        svc.provider.chat.side_effect = RuntimeError("model down")
        cfg.contextual_enrichment = True
        try:
            with patch("lilbee.data.extract.document.get_services", return_value=svc):
                out = _enrich_texts(["chunk text"], "doc head", "a.pdf")
        finally:
            cfg.contextual_enrichment = False
        assert out == [""]
