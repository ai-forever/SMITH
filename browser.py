"""Sync OpenResearcher-style browser page store and SERP/viewport formatting.

Reimplements a small subset of GPT-OSS SimpleBrowser behavior without importing
``gpt_oss``. Formats intentionally match teacher traces so existing
``browser.*`` tool calls remain executable.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

from harness.common import SearchHit

DEFAULT_VIEWPORT_LINES = 50
_LINE_WRAP_WIDTH = 80
_CITATION_CURSOR_RE = re.compile(r"【(\d+)†L\d+(?:-L\d+)?】")


@dataclass
class BrowserPage:
    """One cached browser page (SERP, document, or find results)."""

    cursor: int
    url: str
    title: str
    lines: List[str]
    link_id_to_url: Dict[int, str] = field(default_factory=dict)
    link_id_to_corpus_id: Dict[int, str] = field(default_factory=dict)
    corpus_id: Optional[str] = None
    kind: str = "document"  # serp | document | find


@dataclass
class BrowserPageStore:
    """Session-scoped cursor → page map for ``browser.search/open/find``."""

    pages: Dict[int, BrowserPage] = field(default_factory=dict)
    cursor_to_corpus_id: Dict[int, str] = field(default_factory=dict)
    next_cursor: int = 0
    current_cursor: int = -1

    def allocate_cursor(self) -> int:
        cursor = int(self.next_cursor)
        self.next_cursor = cursor + 1
        return cursor

    def add_page(self, page: BrowserPage) -> BrowserPage:
        self.pages[int(page.cursor)] = page
        self.current_cursor = int(page.cursor)
        corpus_id = str(page.corpus_id or "").strip()
        if corpus_id and page.kind == "document":
            self.cursor_to_corpus_id[int(page.cursor)] = corpus_id
        return page

    def get(self, cursor: int) -> Optional[BrowserPage]:
        return self.pages.get(int(cursor))

    def resolve_cursor(self, raw_cursor: object) -> int:
        try:
            cursor = int(raw_cursor)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            cursor = -1
        if cursor < 0:
            return int(self.current_cursor)
        return cursor


def parse_citation_cursors(explanation: str) -> List[int]:
    """Ordered unique page cursors cited as ``【cursor†L…】`` in explanation."""
    ordered: List[int] = []
    seen: set[int] = set()
    for match in _CITATION_CURSOR_RE.finditer(str(explanation or "")):
        cursor = int(match.group(1))
        if cursor in seen:
            continue
        seen.add(cursor)
        ordered.append(cursor)
    return ordered


def supporting_corpus_ids_from_explanation(
    explanation: str,
    cursor_to_corpus_id: Mapping[int, str],
) -> List[str]:
    """Map citation cursors to corpus ids; drop unknown / SERP-only cursors."""
    supporting: List[str] = []
    seen: set[str] = set()
    for cursor in parse_citation_cursors(explanation):
        corpus_id = str(cursor_to_corpus_id.get(cursor) or "").strip()
        if not corpus_id or corpus_id in seen:
            continue
        seen.add(corpus_id)
        supporting.append(corpus_id)
    return supporting


def _host_from_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        host = urlparse(raw).netloc
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _wrap_text(text: str, *, width: int = _LINE_WRAP_WIDTH) -> List[str]:
    raw = str(text or "")
    if not raw:
        return [""]
    lines: List[str] = []
    for paragraph in raw.splitlines() or [""]:
        if not paragraph:
            lines.append("")
            continue
        start = 0
        while start < len(paragraph):
            lines.append(paragraph[start : start + width])
            start += width
    return lines or [""]


def _viewport_bounds(
    total_lines: int,
    *,
    loc: int,
    num_lines: int,
) -> Tuple[int, int]:
    total = max(0, int(total_lines))
    start = 0 if loc is None or int(loc) < 0 else max(0, int(loc))
    if total == 0:
        return 0, -1
    start = min(start, max(0, total - 1))
    count = DEFAULT_VIEWPORT_LINES if num_lines is None or int(num_lines) < 0 else max(1, int(num_lines))
    end = min(total - 1, start + count - 1)
    return start, end


def render_viewport(page: BrowserPage, *, loc: int = -1, num_lines: int = -1) -> str:
    """Render a teacher-style page viewport for the given line window."""
    start, end = _viewport_bounds(len(page.lines), loc=loc, num_lines=num_lines)
    total = len(page.lines)
    header_title = page.title or page.url or f"cursor={page.cursor}"
    lines_out = [
        f"[{page.cursor}] {header_title} ({page.url})",
        (
            f"**viewing lines [{start} - {end}] of {total}**"
            if total
            else "**viewing lines [0 - -1] of 0**"
        ),
        "",
    ]
    if total and end >= start:
        for idx in range(start, end + 1):
            lines_out.append(f"L{idx}: {page.lines[idx]}")
    return "\n".join(lines_out)


def build_serp_page(
    store: BrowserPageStore,
    *,
    query: str,
    hits: Sequence[SearchHit],
) -> Tuple[BrowserPage, str]:
    """Create a SERP page from FAISS hits and return ``(page, viewport_text)``."""
    cursor = store.allocate_cursor()
    pseudo_url = f"web-search://ts={int(time.time())}"
    title = str(query or "").strip() or "Search Results"
    body: List[str] = ["", f"URL: {pseudo_url}", "# Search Results", ""]
    link_id_to_url: Dict[int, str] = {}
    link_id_to_corpus_id: Dict[int, str] = {}

    for link_id, hit in enumerate(hits):
        corpus_id = str(hit.corpus_id or "").strip()
        url = str((hit.metadata or {}).get("url") or "").strip()
        if not url and corpus_id:
            url = f"corpus://{corpus_id}"
        host = _host_from_url(url) or "corpus"
        marker = f"【{link_id}†Doc {corpus_id or link_id}†{host}】"
        title_text = str(hit.title or "").strip()
        snippet = str(hit.text or "").replace("\n", " ").strip()
        if title_text and snippet:
            if snippet.startswith(title_text):
                display = f"  * {marker} {snippet}"
            else:
                display = f"  * {marker} {title_text} {snippet}"
        elif title_text:
            display = f"  * {marker} {title_text}"
        else:
            display = f"  * {marker} {snippet}"
        body.extend(_wrap_text(display))
        link_id_to_url[link_id] = url
        if corpus_id:
            link_id_to_corpus_id[link_id] = corpus_id

    page = BrowserPage(
        cursor=cursor,
        url=pseudo_url,
        title=title,
        lines=body,
        link_id_to_url=link_id_to_url,
        link_id_to_corpus_id=link_id_to_corpus_id,
        corpus_id=None,
        kind="serp",
    )
    store.add_page(page)
    return page, render_viewport(page)


def build_document_page(
    store: BrowserPageStore,
    *,
    corpus_id: str,
    text: str,
    url: str = "",
    title: str = "",
) -> Tuple[BrowserPage, str]:
    """Cache a full document and return the default viewport."""
    cursor = store.allocate_cursor()
    doc_id = str(corpus_id or "").strip() or "unknown"
    page_url = str(url or "").strip() or f"corpus://{doc_id}"
    page_title = str(title or "").strip() or f"Doc {doc_id}"
    body: List[str] = ["", f"URL: {page_url}"]
    body.extend(_wrap_text(str(text or "")))
    page = BrowserPage(
        cursor=cursor,
        url=page_url,
        title=page_title if page_title.startswith("Doc ") else f"Doc {doc_id}",
        lines=body,
        corpus_id=doc_id,
        kind="document",
    )
    # Preserve a human title in the header when provided.
    if title and not str(title).strip().startswith("Doc "):
        page.title = f"Doc {doc_id}"
    store.add_page(page)
    return page, render_viewport(page)


def build_find_page(
    store: BrowserPageStore,
    *,
    source: BrowserPage,
    pattern: str,
) -> Tuple[BrowserPage, str]:
    """Scan ``source`` lines for exact substring matches; return a find page."""
    needle = str(pattern or "")
    matches: List[Tuple[int, str]] = []
    if needle:
        for idx, line in enumerate(source.lines):
            if needle in line:
                matches.append((idx, line))

    cursor = store.allocate_cursor()
    doc_label = source.title or f"cursor={source.cursor}"
    find_url = f"{source.url}/find?pattern={needle}"
    header = f"Find results for text: `{needle}` in `{doc_label}` ({find_url})"
    body: List[str] = []
    if not matches:
        body = ["", f"No matches for `{needle}`."]
    else:
        for match_id, (line_idx, line) in enumerate(matches):
            body.append(f"# 【{match_id}†match at L{line_idx}】")
            # Include a small window around the match line.
            start = max(0, line_idx - 1)
            end = min(len(source.lines) - 1, line_idx + 2)
            for around in range(start, end + 1):
                body.append(source.lines[around])
            body.append("")

    page = BrowserPage(
        cursor=cursor,
        url=find_url,
        title=header,
        lines=body,
        corpus_id=source.corpus_id,
        kind="find",
    )
    store.add_page(page)
    # Find pages inherit corpus_id for citation mapping when source was a document.
    if source.corpus_id:
        store.cursor_to_corpus_id[cursor] = str(source.corpus_id)
    return page, render_viewport(page)


def coerce_int(value: object, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
