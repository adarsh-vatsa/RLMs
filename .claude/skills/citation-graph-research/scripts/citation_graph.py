#!/usr/bin/env python3
"""Walk the citation graph around a seed paper (stdlib only).

Pulls forward citations (papers that cite the seed) and/or backward references
from Semantic Scholar, falling back to OpenAlex. Ranks results by topic-keyword
hits, the Semantic Scholar "influential" flag, recency, and citation count, and
prints a markdown table (or JSON).

Examples:
    python citation_graph.py arXiv:2512.24601 --keywords cache memo reuse
    python citation_graph.py arXiv:2512.24601 --direction references --top 30
    python citation_graph.py "Recursive Language Models" --json > rlm_citers.json

Set SEMANTIC_SCHOLAR_API_KEY for higher rate limits (optional).
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

S2 = "https://api.semanticscholar.org/graph/v1"
OA = "https://api.openalex.org"
S2_FIELDS = "title,year,venue,citationCount,externalIds,abstract,authors,publicationDate"
USER_AGENT = "citation-graph-research/1.0 (research tooling)"


def _get(url, headers=None, retries=3):
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdrs), timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise
    raise RuntimeError(f"unreachable: {url}")


def _s2_headers():
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    return {"x-api-key": key} if key else {}


def _normalize_seed(seed):
    """Accept arXiv IDs/URLs, DOIs, S2 IDs, or free-text titles."""
    seed = seed.strip()
    m = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", seed)
    if m:
        return f"arXiv:{m.group(1)}"
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", seed):
        return f"arXiv:{seed.split('v')[0]}"
    if seed.startswith("10.") and "/" in seed:
        return f"DOI:{seed}"
    return seed


# ---------------------------------------------------------------- Semantic Scholar
def s2_resolve(seed):
    seed = _normalize_seed(seed)
    if re.match(r"^(arXiv|DOI|CorpusId|PMID|ACL|MAG|URL):", seed) or re.fullmatch(r"[0-9a-f]{40}", seed):
        return _get(f"{S2}/paper/{urllib.parse.quote(seed, safe=':')}?fields={S2_FIELDS}", _s2_headers())
    q = urllib.parse.urlencode({"query": seed, "limit": 1, "fields": S2_FIELDS})
    hits = _get(f"{S2}/paper/search?{q}", _s2_headers()).get("data") or []
    if not hits:
        raise LookupError(f"No Semantic Scholar match for {seed!r}")
    return hits[0]


def s2_edges(paper_id, direction, limit):
    key = "citingPaper" if direction == "citations" else "citedPaper"
    fields = f"isInfluential,contexts,intents,{','.join(f'{key}.{f}' for f in S2_FIELDS.split(','))}"
    out, offset = [], 0
    while len(out) < limit:
        page = min(100, limit - len(out))
        url = f"{S2}/paper/{paper_id}/{direction}?fields={fields}&limit={page}&offset={offset}"
        data = _get(url, _s2_headers())
        for row in data.get("data") or []:
            p = row.get(key) or {}
            if not p.get("title"):
                continue
            out.append({
                "title": p["title"],
                "year": p.get("year"),
                "date": p.get("publicationDate"),
                "venue": p.get("venue") or "",
                "citations": p.get("citationCount") or 0,
                "authors": ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:3]),
                "abstract": p.get("abstract") or "",
                "contexts": row.get("contexts") or [],
                "intents": row.get("intents") or [],
                "influential": bool(row.get("isInfluential")),
                "url": _paper_url(p.get("externalIds") or {}, p.get("paperId")),
            })
        if data.get("next") is None:
            break
        offset = data["next"]
        time.sleep(1.0)  # stay under the unauthenticated rate limit
    return out


def _paper_url(ext, s2_id):
    if ext.get("ArXiv"):
        return f"https://arxiv.org/abs/{ext['ArXiv']}"
    if ext.get("DOI"):
        return f"https://doi.org/{ext['DOI']}"
    return f"https://www.semanticscholar.org/paper/{s2_id}" if s2_id else ""


# ---------------------------------------------------------------- OpenAlex fallback
def _oa_abstract(inv):
    if not inv:
        return ""
    words = sorted((pos, w) for w, positions in inv.items() for pos in positions)
    return " ".join(w for _, w in words)


def oa_resolve(seed):
    seed = _normalize_seed(seed)
    if seed.startswith("arXiv:"):
        return _get(f"{OA}/works/doi:10.48550/arXiv.{seed[6:]}")
    if seed.startswith("DOI:"):
        return _get(f"{OA}/works/doi:{seed[4:]}")
    hits = _get(f"{OA}/works?{urllib.parse.urlencode({'search': seed, 'per-page': 1})}").get("results") or []
    if not hits:
        raise LookupError(f"No OpenAlex match for {seed!r}")
    return hits[0]


def oa_edges(work, direction, limit):
    wid = work["id"].rsplit("/", 1)[-1]
    flt = f"cites:{wid}" if direction == "citations" else f"cited_by:{wid}"
    out, cursor = [], "*"
    while cursor and len(out) < limit:
        q = urllib.parse.urlencode({"filter": flt, "per-page": min(200, limit - len(out)), "cursor": cursor})
        data = _get(f"{OA}/works?{q}")
        for w in data.get("results") or []:
            out.append({
                "title": w.get("title") or "",
                "year": w.get("publication_year"),
                "date": w.get("publication_date"),
                "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
                "citations": w.get("cited_by_count") or 0,
                "authors": ", ".join(a["author"]["display_name"] for a in (w.get("authorships") or [])[:3]),
                "abstract": _oa_abstract(w.get("abstract_inverted_index")),
                "contexts": [], "intents": [], "influential": False,
                "url": w.get("doi") or w.get("id"),
            })
        cursor = (data.get("meta") or {}).get("next_cursor")
    return out


# ---------------------------------------------------------------- ranking
def score(paper, keywords):
    text = " ".join([paper["title"], paper["abstract"], " ".join(paper["contexts"])]).lower()
    hits = sorted({k for k in keywords if k.lower() in text})
    paper["keyword_hits"] = hits
    recency = max(0, (paper.get("year") or 0) - 2020)
    return (len(hits) * 10) + (5 if paper["influential"] else 0) + recency + min(paper["citations"], 200) ** 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seed", help="arXiv ID/URL, DOI, Semantic Scholar ID, or paper title")
    ap.add_argument("--direction", choices=("citations", "references", "both"), default="citations")
    ap.add_argument("--keywords", nargs="*", default=[], help="topic terms used to rank and filter")
    ap.add_argument("--only-matching", action="store_true", help="drop papers with zero keyword hits")
    ap.add_argument("--limit", type=int, default=500, help="max edges fetched per direction")
    ap.add_argument("--top", type=int, default=40, help="rows to print")
    ap.add_argument("--source", choices=("auto", "s2", "openalex"), default="auto")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = ap.parse_args()

    directions = ["citations", "references"] if args.direction == "both" else [args.direction]
    source, seed_meta, results = None, None, []
    errors = []
    for src in (["s2", "openalex"] if args.source == "auto" else [args.source]):
        try:
            if src == "s2":
                seed_meta = s2_resolve(args.seed)
                for d in directions:
                    results += [dict(p, direction=d) for p in s2_edges(seed_meta["paperId"], d, args.limit)]
            else:
                seed_meta = oa_resolve(args.seed)
                for d in directions:
                    results += [dict(p, direction=d) for p in oa_edges(seed_meta, d, args.limit)]
            source = src
            break
        except Exception as e:  # network block, 404, rate limit: try the next source
            errors.append(f"{src}: {e}")
            results = []
    if source is None:
        sys.exit("All sources failed:\n  " + "\n  ".join(errors) +
                 "\nIf this is a network/proxy block, allow api.semanticscholar.org and api.openalex.org.")

    for p in results:
        p["score"] = round(score(p, args.keywords), 2)
    if args.only_matching and args.keywords:
        results = [p for p in results if p["keyword_hits"]]
    results.sort(key=lambda p: p["score"], reverse=True)

    title = seed_meta.get("title") or seed_meta.get("display_name")
    if args.json:
        json.dump({"seed": title, "source": source, "retrieved": time.strftime("%Y-%m-%d"),
                   "count": len(results), "papers": results[:args.top]}, sys.stdout, indent=2)
        return
    print(f"# Citation graph: {title}\n")
    print(f"Source: {source} · retrieved {time.strftime('%Y-%m-%d')} · {len(results)} edges"
          f" · directions: {', '.join(directions)} · keywords: {', '.join(args.keywords) or '-'}\n")
    print("| # | Dir | Year | Title | Venue | Cites | Infl. | Keyword hits | URL |")
    print("|---|---|---|---|---|---|---|---|---|")
    for i, p in enumerate(results[:args.top], 1):
        t = p["title"].replace("|", "/")
        print(f"| {i} | {p['direction'][:4]} | {p.get('year') or ''} | {t} | {p['venue'][:30]} | "
              f"{p['citations']} | {'Y' if p['influential'] else ''} | {', '.join(p['keyword_hits'])} | {p['url']} |")
    ctx = [p for p in results[:args.top] if p["contexts"] and p["keyword_hits"]]
    if ctx:
        print("\n## How matching papers cite the seed\n")
        for p in ctx[:15]:
            print(f"- **{p['title']}** ({p.get('year')}): \"{p['contexts'][0].strip()[:300]}\"")


if __name__ == "__main__":
    main()
