"""Conservative, free, multi-source team-news collection.

The collector is deliberately evidence-first.  It searches two independent RSS
indexes, records every relevant article with its publisher and timestamp, and
only changes model inputs when an explicit player/status claim is corroborated
by two publishers.  A single report is retained for review but cannot remove a
player or alter the match model.
"""
from __future__ import annotations

import datetime as dt
import email.utils
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from leagues import fixtures as fixture_feed
from leagues.config import LEAGUES


ROOT = Path(__file__).resolve().parent.parent
BEST_PATH = ROOT / "data" / "leagues" / "best.json"
NEWS_PATH = ROOT / "data-raw" / "leagues" / "news.json"
LEAGUE_PATH = ROOT / "data" / "leagues"

SEARCH_FEEDS = {
    "google-news": (
        "https://news.google.com/rss/search?q={query}"
        "&hl=en-GB&gl=GB&ceid=GB:en"
    ),
    "bing-news": "https://www.bing.com/news/search?q={query}&format=rss",
}
OUT_WORDS = re.compile(
    r"\b(ruled out|will miss|set to miss|suspended|unavailable|"
    r"out for|out of the|misses? the)\b", re.I)
DOUBT_WORDS = re.compile(
    r"\b(doubtful|fitness test|could miss|may miss|unlikely to (start|play)|"
    r"injury doubt|race against time)\b", re.I)
RECENT_HOURS = 96
WINDOW_HOURS = 72


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _published(value: str, fallback: dt.datetime) -> dt.datetime:
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except (TypeError, ValueError):
        return fallback


def fetch_rss(url: str, timeout: int = 15) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "worldcup-team-news/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_rss(payload: bytes, feed: str, now: dt.datetime) -> list[dict]:
    root = ET.fromstring(payload)
    items = []
    for node in root.findall(".//item"):
        title = (node.findtext("title") or "").strip()
        link = (node.findtext("link") or "").strip()
        published = _published(node.findtext("pubDate") or "", now)
        # Google uses <source>; Bing uses the namespaced <News:Source>. Resolve
        # both to the ORIGINAL publisher. Treating "bing.com" as the publisher
        # would let one newspaper indexed by Google and Bing count twice.
        source_node = next(
            (child for child in node
             if child.tag.rsplit("}", 1)[-1].casefold() == "source"), None)
        publisher = ((source_node.text or "").strip() if source_node is not None
                     else urllib.parse.urlparse(link).netloc)
        if title and link:
            items.append({
                "title": title,
                "url": link,
                "publisher": publisher or feed,
                "published_at": _iso(published),
                "feed": feed,
            })
    return items


def upcoming_best(path: Path = BEST_PATH, now: dt.datetime | None = None) -> list[dict]:
    now = now or utcnow()
    if not path.exists():
        return []
    fixtures = json.loads(path.read_text(encoding="utf-8")).get("upcoming", [])
    result = []
    for fixture in fixtures:
        try:
            kickoff = dt.datetime.fromisoformat(
                str(fixture["date"]).replace("Z", "+00:00"))
            if kickoff.tzinfo is None:
                kickoff = kickoff.replace(tzinfo=dt.timezone.utc)
        except (KeyError, TypeError, ValueError):
            continue
        hours = (kickoff - now).total_seconds() / 3600
        if 0 < hours <= WINDOW_HOURS:
            result.append(fixture)
    return result


def upcoming_fixtures(now: dt.datetime | None = None,
                      loader=fixture_feed.fetch_fixtures) -> list[dict]:
    """Every unplayed league fixture inside the news window.

    Discovery must not read the previous ``best.json``. A model refresh can add a
    new Best Pick, and researching only the old board creates a loop where the
    freshness gate rejects the new fixture before it can ever be committed.
    Reading the canonical fixture feed makes news coverage a superset of any board
    the subsequent model publish can produce.
    """
    now = now or utcnow()
    result = []
    for league in LEAGUES:
        frame = loader(league)
        for fixture in frame.to_dict("records"):
            if fixture.get("played"):
                continue
            try:
                kickoff = dt.datetime.fromisoformat(
                    str(fixture["date"]).replace("Z", "+00:00"))
                if kickoff.tzinfo is None:
                    kickoff = kickoff.replace(tzinfo=dt.timezone.utc)
            except (KeyError, TypeError, ValueError):
                continue
            hours = (kickoff - now).total_seconds() / 3600
            if 0 < hours <= WINDOW_HOURS:
                result.append({
                    "league_key": league,
                    "date": _iso(kickoff),
                    "home": fixture["home"],
                    "away": fixture["away"],
                })
    return result


def player_candidates(league: str, team: str,
                      league_dir: Path = LEAGUE_PATH) -> list[str]:
    path = league_dir / f"{league.lower()}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = {
        prop["player"]
        for match in payload.get("matches", [])
        for prop in match.get("props", [])
        if prop.get("team") == team and prop.get("player")
    }
    return sorted(names, key=lambda name: (-len(name), name))


def classify(title: str, players: list[str]) -> tuple[str, str] | None:
    """Match a candidate's full name as a whole word/phrase, not a bare substring.

    A surname can itself be a common English-word fragment: Son Heung-min's
    surname is inside "season"/"reason". `"son" in title.casefold()` matched
    both, misclassifying unrelated headlines (a manager's suspension, a generic
    pundit-doubt line) as Son being out/doubtful. `\\b...\\b` still finds the name
    anywhere in the title, it just requires it to start and end on a word
    boundary, so it can no longer match mid-word.
    """
    player = next(
        (name for name in players if re.search(rf"\b{re.escape(name)}\b", title, re.I)),
        None)
    if not player:
        return None
    if OUT_WORDS.search(title):
        return player, "out"
    if DOUBT_WORDS.search(title):
        return player, "doubt"
    return None


_GENERIC_PUBLISHER_TOKENS = {"the", "on", "msn", "news", "co", "com", "uk"}


def _publisher_tokens(evidence: dict) -> frozenset[str]:
    """Tokens identifying a publisher, with generic boilerplate words stripped.

    Kept as a TOKEN SET rather than one concatenated string so a regional-prefix
    syndication byline can still be recognised as the same outlet (see
    `_same_publisher`): exact-string keys treated "Evening Standard" and "London
    Evening Standard on MSN" as two different sources, when the second is the
    first republished under an aggregator's local-paper byline.
    """
    value = evidence.get("publisher") or evidence.get("url") or ""
    host = urllib.parse.urlparse(value).netloc or value
    text = re.sub(r"[^a-z0-9]+", " ", host.casefold().removeprefix("www."))
    tokens = {t for t in text.split() if t and t not in _GENERIC_PUBLISHER_TOKENS}
    return frozenset(tokens) or frozenset({text})


def _same_publisher(a: frozenset[str], b: frozenset[str]) -> bool:
    """Same real-world outlet if the smaller name's tokens are fully contained in
    the larger name's -- catching a regional prefix added by syndication -- but
    ONLY when that overlap is at least two words. A single generic word ("Times",
    "Sport") is shared by too many unrelated real outlets (Times of India, NY
    Times, Sunday Times...) for a bare one-token subset match to prove identity;
    requiring two lets "Evening Standard" match its longer variant while still
    keeping "Times" and "Sunday Times" apart.
    """
    if a == b:
        return True
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    return len(small) >= 2 and small <= big


def corroborated(evidence: list[dict], status: str) -> set[str]:
    by_player: dict[str, list[frozenset[str]]] = {}
    for item in evidence:
        if item.get("status") != status:
            continue
        clusters = by_player.setdefault(item["player"], [])
        tokens = _publisher_tokens(item)
        if not any(_same_publisher(tokens, c) for c in clusters):
            clusters.append(tokens)
    return {player for player, clusters in by_player.items() if len(clusters) >= 2}


def collect_team(team: str, opponent: str, players: list[str], fetcher=fetch_rss,
                 now: dt.datetime | None = None) -> tuple[list[dict], list[str]]:
    now = now or utcnow()
    query = urllib.parse.quote(
        f'"{team}" {opponent} injury OR suspended OR "ruled out" OR doubtful')
    evidence, successful = [], []
    cutoff = now - dt.timedelta(hours=RECENT_HOURS)
    seen = set()
    for feed, template in SEARCH_FEEDS.items():
        try:
            items = parse_rss(fetcher(template.format(query=query)), feed, now)
            successful.append(feed)
        except Exception as exc:
            print(f"WARNING: {feed} team-news search failed for {team}: {exc}")
            continue
        for item in items:
            published = dt.datetime.fromisoformat(
                item["published_at"].replace("Z", "+00:00"))
            if published < cutoff:
                continue
            finding = classify(item["title"], players)
            if not finding:
                continue
            key = (item["url"], finding)
            if key in seen:
                continue
            seen.add(key)
            player, status = finding
            evidence.append({**item, "player": player, "status": status})
    return evidence, successful


def refresh(news_path: Path = NEWS_PATH, best_path: Path | None = None,
            league_dir: Path = LEAGUE_PATH, fetcher=fetch_rss,
            now: dt.datetime | None = None, fixture_loader=None) -> dict:
    now = now or utcnow()
    base = (json.loads(news_path.read_text(encoding="utf-8"))
            if news_path.exists() else {})
    # ``best_path`` remains as an explicit test/backfill seam. Production omits it
    # and researches ALL imminent fixtures, so a newly generated Best Pick can
    # never be absent merely because it was not on the previous board.
    imminent = (upcoming_best(best_path, now) if best_path is not None
                else upcoming_fixtures(now, fixture_loader or fixture_feed.fetch_fixtures))
    for fixture in imminent:
        league = fixture["league_key"]
        section = base.setdefault(league, {})
        for team, opponent in ((fixture["home"], fixture["away"]),
                               (fixture["away"], fixture["home"])):
            candidates = player_candidates(league, team, league_dir)
            evidence, successful = collect_team(
                team, opponent, candidates, fetcher=fetcher, now=now)
            current = section.get(team) or {}
            # Preserve human/official confirmed inputs. Automation owns only the
            # players recorded in its previous metadata block.
            previous = current.get("automation") or {}
            old_auto_out = set(previous.get("out") or [])
            old_auto_doubt = set(previous.get("doubt") or [])
            manual_out = set(current.get("out") or []) - old_auto_out
            manual_doubt = set(current.get("doubt") or []) - old_auto_doubt
            auto_out = corroborated(evidence, "out")
            auto_doubt = corroborated(evidence, "doubt") - auto_out
            entry = {
                **current,
                "out": sorted(manual_out | auto_out),
                "doubt": sorted((manual_doubt | auto_doubt) - auto_out),
                "automation": {
                    "checked_at": _iso(now),
                    "feeds_ok": successful,
                    "evidence": evidence,
                    "out": sorted(auto_out),
                    "doubt": sorted(auto_doubt),
                    "policy": "two-independent-publishers",
                },
            }
            # A successful check requires both independent indexes. A partial
            # outage remains visible and cannot falsely satisfy the freshness gate.
            if len(set(successful)) == len(SEARCH_FEEDS):
                entry["checked"] = _iso(now)
            section[team] = entry
    base["_verified_on"] = (now.date().isoformat()
                            if imminent else base.get("_verified_on"))
    return base


def write_refreshed(**kwargs) -> dict:
    result = refresh(**kwargs)
    NEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEWS_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    return result
