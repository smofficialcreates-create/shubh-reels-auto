"""Free, keyless research: what is current, and what an article actually says.

Nothing here needs an API key. Every network call fails soft: a source that cannot be
reached is skipped, and with no sources at all the pipeline falls back to evergreen mode.
"""
from __future__ import annotations

import datetime as dt
import email.utils
import html
import json
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests

from . import safety

UA = {"User-Agent": "Mozilla/5.0 (compatible; ShubhReelBot/1.0)", "Accept-Language": "en"}
MIN_ARTICLE_CHARS = 700
MAX_EXCERPT_CHARS = 4200


@dataclass
class Candidate:
    id: int
    title: str
    url: str
    publisher: str
    published: str          # ISO date
    age_days: float
    snippet: str = ""
    points: int = 0

    def brief(self) -> dict:
        d = asdict(self)
        d["snippet"] = d["snippet"][:240]
        return d


def _log(msg: str):
    print(f"[research] {msg}", file=sys.stderr)


def load_sources(path: str = "data/sources.json") -> dict:
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return {"hn": [], "feeds": [], "skip_hosts": [], "max_age_days": 10}


def _get(url: str, timeout: int = 20, **kw) -> requests.Response | None:
    try:
        r = requests.get(url, headers=UA, timeout=timeout, **kw)
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        _log(f"skipped {urlparse(url).netloc}: {type(e).__name__}")
        return None


def _strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def _parse_date(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    try:
        d = email.utils.parsedate_to_datetime(s)
    except (TypeError, ValueError):
        try:
            d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def parse_feed(xml_text: str, publisher: str) -> list[dict]:
    """RSS 2.0 or Atom -> [{title, url, published(datetime), snippet}]."""
    out = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for it in root.iter("item"):
        out.append({"title": _strip_tags(it.findtext("title", "")), "url": (it.findtext("link") or "").strip(),
                    "published": _parse_date(it.findtext("pubDate") or it.findtext("{http://purl.org/dc/elements/1.1/}date")),
                    "snippet": _strip_tags(it.findtext("description", "")), "publisher": publisher})
    for it in root.findall("a:entry", ns):
        link = it.find("a:link[@rel='alternate']", ns) or it.find("a:link", ns)
        out.append({"title": _strip_tags(it.findtext("a:title", "", ns)), "url": (link.get("href") if link is not None else ""),
                    "published": _parse_date(it.findtext("a:published", None, ns) or it.findtext("a:updated", None, ns)),
                    "snippet": _strip_tags(it.findtext("a:summary", "", ns)), "publisher": publisher})
    return out


def parse_hn(data: dict) -> list[dict]:
    out = []
    for h in data.get("hits", []):
        if not h.get("url") or not h.get("title"):
            continue
        out.append({"title": h["title"], "url": h["url"], "points": int(h.get("points") or 0),
                    "published": dt.datetime.fromtimestamp(h.get("created_at_i", 0), dt.timezone.utc),
                    "snippet": "", "publisher": urlparse(h["url"]).netloc.removeprefix("www.")})
    return out


def gather(now: dt.datetime | None = None, sources: dict | None = None) -> list[Candidate]:
    """Fresh, de-duplicated candidates from every reachable source."""
    now = now or dt.datetime.now(dt.timezone.utc)
    src = sources or load_sources()
    raw: list[dict] = []

    def one_feed(item):
        name, url = item
        r = _get(url)
        return parse_feed(r.text, name) if r is not None else []

    def one_hn(url):
        r = _get(url)
        try:
            return parse_hn(r.json()) if r is not None else []
        except ValueError:
            return []

    with ThreadPoolExecutor(6) as ex:
        for res in ex.map(one_feed, src.get("feeds", [])):
            raw += res
        for res in ex.map(one_hn, src.get("hn", [])):
            raw += res
    skip = tuple(src.get("skip_hosts", []))
    max_age = float(src.get("max_age_days", 10))
    cands, seen_titles, seen_urls = [], [], set()
    for c in sorted(raw, key=lambda c: -(c.get("points", 0))):
        host = urlparse(c["url"]).netloc.lower().removeprefix("www.")
        if not c["url"].startswith("http") or any(host == s or host.endswith("." + s) for s in skip):
            continue
        if c["published"] is None:
            continue
        age = (now - c["published"]).total_seconds() / 86400
        if age < -0.2 or age > max_age or len(c["title"]) < 12:
            continue
        if c["url"] in seen_urls or safety.too_similar(c["title"], seen_titles, 0.6):
            continue
        seen_urls.add(c["url"])
        seen_titles.append(c["title"])
        cands.append(Candidate(len(cands) + 1, c["title"], c["url"], c["publisher"], c["published"].date().isoformat(),
                               round(max(age, 0), 1), c.get("snippet", ""), int(c.get("points", 0))))
    cands.sort(key=lambda c: (c.age_days, -c.points))
    return cands[:30]


# ---------------------------------------------------------------------------- article text
class _Article(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "aside", "header", "form", "noscript", "figure", "svg", "button"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.in_p = False
        self.in_article = 0
        self.buf: list[str] = []
        self.paras_all: list[str] = []
        self.paras_article: list[str] = []
        self.og_title = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "meta" and a.get("property") == "og:title":
            self.og_title = a.get("content", "")
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "article":
            self.in_article += 1
        elif tag in ("p", "li") and not self.skip:
            self.in_p = True
            self.buf = []

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "article":
            self.in_article = max(0, self.in_article - 1)
        elif tag in ("p", "li") and self.in_p:
            self.in_p = False
            t = re.sub(r"\s+", " ", "".join(self.buf)).strip()
            if len(t) >= 45:
                self.paras_all.append(t)
                if self.in_article:
                    self.paras_article.append(t)

    def handle_data(self, data):
        if self.in_p and not self.skip:
            self.buf.append(data)


def extract_text(page_html: str) -> tuple[str, str]:
    p = _Article()
    try:
        p.feed(page_html)
    except Exception:  # noqa: BLE001 - malformed HTML: use what was collected
        pass
    paras = p.paras_article if sum(map(len, p.paras_article)) >= MIN_ARTICLE_CHARS else p.paras_all
    seen, uniq = set(), []
    for t in paras:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return p.og_title.strip(), "\n".join(uniq)


def fetch_article(url: str) -> dict | None:
    """{'title','text'} or None when the page is unreachable, not HTML, or too thin to trust."""
    r = _get(url, timeout=25, stream=True)
    if r is None:
        return None
    if "html" not in r.headers.get("content-type", "html").lower():
        return None
    body = r.raw.read(2_000_000, decode_content=True)
    title, text = extract_text(body.decode(r.encoding or "utf-8", errors="replace"))
    if len(text) < MIN_ARTICLE_CHARS:
        _log(f"too little article text at {urlparse(url).netloc}")
        return None
    return {"title": title, "text": text[:MAX_EXCERPT_CHARS]}
