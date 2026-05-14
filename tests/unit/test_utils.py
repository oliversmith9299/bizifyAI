"""
Unit tests for agents/utils.py
================================
Tests pure logic only — all HTTP calls are mocked.
No database or LLM connections are made.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.utils import (
    extract_sources,
    fetch_reddit,
    fetch_web,
    parse_llm_json,
    search_serper,
    truncate_sources,
)


# ── parse_llm_json ────────────────────────────────────────────────────────────

class TestParseLlmJson:
    def test_plain_json_string(self):
        raw = '{"key": "value", "num": 42}'
        result = parse_llm_json(raw)
        assert result == {"key": "value", "num": 42}

    def test_json_with_code_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        result = parse_llm_json(raw)
        assert result == {"key": "value"}

    def test_json_with_plain_code_fence(self):
        raw = '```\n{"key": "value"}\n```'
        result = parse_llm_json(raw)
        assert result == {"key": "value"}

    def test_json_embedded_in_prose(self):
        raw = 'Here is the analysis:\n{"key": "value"}\nThat is all.'
        result = parse_llm_json(raw)
        assert result == {"key": "value"}

    def test_json_array(self):
        raw = '[{"id": 1}, {"id": 2}]'
        result = parse_llm_json(raw)
        assert result == [{"id": 1}, {"id": 2}]

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            parse_llm_json("")

    def test_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            parse_llm_json("   \n  ")

    def test_none_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            parse_llm_json(None)

    def test_garbage_string_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_llm_json("this is not json at all, no brackets")

    def test_nested_json(self):
        raw = '{"outer": {"inner": [1, 2, 3]}}'
        result = parse_llm_json(raw)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_code_fence_with_extra_whitespace(self):
        raw = '```json\n\n  {"a": 1}  \n\n```'
        result = parse_llm_json(raw)
        assert result == {"a": 1}


# ── truncate_sources ──────────────────────────────────────────────────────────

class TestTruncateSources:
    def _make_sources(self, n: int) -> list:
        return [{"url": f"https://example.com/{i}", "content": "x" * 100} for i in range(n)]

    def test_keeps_all_when_under_limit(self):
        sources = self._make_sources(3)
        result = truncate_sources(sources, max_chars=100_000)
        assert len(result) == 3

    def test_drops_from_tail_until_fits(self):
        # create sources large enough that trimming is required
        sources = [{"url": f"u{i}", "content": "y" * 10_000} for i in range(20)]
        result = truncate_sources(sources, max_chars=50_000)
        assert len(result) < 20

    def test_never_drops_below_two(self):
        sources = [{"url": f"u{i}", "content": "z" * 50_000} for i in range(10)]
        result = truncate_sources(sources, max_chars=1)
        assert len(result) >= 2

    def test_empty_list_returns_empty(self):
        assert truncate_sources([], max_chars=1_000) == []

    def test_single_source_stays(self):
        sources = [{"url": "u1", "content": "hello"}]
        result = truncate_sources(sources, max_chars=1)
        assert len(result) == 1


# ── extract_sources ───────────────────────────────────────────────────────────

class TestExtractSources:
    def _serper_response(self, urls: list) -> dict:
        return {
            "organic": [
                {"link": url, "title": f"Title {i}", "snippet": f"Snippet {i}"}
                for i, url in enumerate(urls)
            ]
        }

    def test_extracts_url_title_snippet(self):
        resp = self._serper_response(["https://example.com/page"])
        sources = extract_sources(resp)
        assert len(sources) == 1
        assert sources[0]["url"] == "https://example.com/page"
        assert sources[0]["title"] == "Title 0"

    def test_skips_entries_without_url(self):
        resp = {"organic": [{"title": "No URL here", "snippet": "..."}]}
        sources = extract_sources(resp)
        assert sources == []

    def test_marks_reddit_urls(self):
        resp = self._serper_response(["https://www.reddit.com/r/test/comments/abc"])
        sources = extract_sources(resp)
        assert sources[0]["type"] == "reddit"

    def test_marks_non_reddit_as_web(self):
        resp = self._serper_response(["https://techcrunch.com/article"])
        sources = extract_sources(resp)
        assert sources[0]["type"] == "web"

    def test_empty_response_returns_empty_list(self):
        assert extract_sources({}) == []

    def test_multiple_sources_preserved_in_order(self):
        urls = [f"https://example.com/{i}" for i in range(5)]
        resp = self._serper_response(urls)
        sources = extract_sources(resp)
        assert [s["url"] for s in sources] == urls


# ── fetch_web ─────────────────────────────────────────────────────────────────

class TestFetchWeb:
    def test_returns_paragraph_text(self):
        fake_html = "<html><body><p>Hello world, this is a paragraph with enough chars.</p></body></html>"
        mock_resp = MagicMock()
        mock_resp.text = fake_html
        mock_resp.raise_for_status = MagicMock()

        with patch("agents.utils.requests.get", return_value=mock_resp):
            result = fetch_web("https://example.com")

        assert "Hello world" in result

    def test_returns_fallback_on_error(self):
        with patch("agents.utils.requests.get", side_effect=Exception("timeout")):
            result = fetch_web("https://example.com", fallback="fallback text")

        assert result == "fallback text"

    def test_truncates_to_content_chars_limit(self):
        long_text = "word " * 1000
        fake_html = f"<html><body><p>{long_text}</p></body></html>"
        mock_resp = MagicMock()
        mock_resp.text = fake_html
        mock_resp.raise_for_status = MagicMock()

        with patch("agents.utils.requests.get", return_value=mock_resp):
            result = fetch_web("https://example.com")

        assert len(result) <= 800


# ── fetch_reddit ──────────────────────────────────────────────────────────────

class TestFetchReddit:
    def test_non_comment_url_returns_empty_string(self):
        result = fetch_reddit("https://reddit.com/r/startups/")
        assert result == ""

    def test_comment_url_parses_post_and_comments(self):
        fake_data = [
            {"data": {"children": [{"data": {"selftext": "Post body"}}]}},
            {"data": {"children": [
                {"data": {"body": "Comment 1"}},
                {"data": {"body": "Comment 2"}},
            ]}},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_data

        with patch("agents.utils.requests.get", return_value=mock_resp):
            result = fetch_reddit("https://reddit.com/r/test/comments/abc/title/")

        assert "Post body" in result
        assert "Comment 1" in result

    def test_returns_empty_on_exception(self):
        with patch("agents.utils.requests.get", side_effect=Exception("network error")):
            result = fetch_reddit("https://reddit.com/r/test/comments/abc/")

        assert result == ""


# ── search_serper ─────────────────────────────────────────────────────────────

class TestSearchSerper:
    def test_returns_parsed_response(self):
        fake_payload = {"organic": [{"link": "https://example.com", "title": "Test"}]}
        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_payload
        mock_resp.raise_for_status = MagicMock()

        with patch("agents.utils.requests.post", return_value=mock_resp):
            result = search_serper("test query", api_key="fake-key")

        assert result == fake_payload

    def test_returns_empty_dict_on_error(self):
        with patch("agents.utils.requests.post", side_effect=Exception("connection error")):
            result = search_serper("test query", api_key="fake-key")

        assert result == {}
