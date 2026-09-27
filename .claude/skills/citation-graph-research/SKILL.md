---
name: citation-graph-research
description: Literature research by walking the citation graph from a seed paper instead of open-ended web search. Use whenever a research question concerns a specific paper, method, or line of work (e.g. "what has happened since RLMs?", "who else memoizes sub-calls?", "is X novel?", related-work or positioning questions).
---

# Citation-graph research

Anyone who works seriously on an idea cites its key paper. So for questions about
a specific paper or line of work, the citation graph is a better search index than
keyword web search. It is higher-recall for true follow-ups, lower-noise, and
every hit comes with a verifiable link.

## Workflow

1. **Pick seeds.** Identify 1–3 seed papers: the paper the question names, plus
   the closest paper from any adjacent literature the question touches. For
   example, use RLM (`arXiv:2512.24601`) plus a semantic-caching paper when
   asking about memoizing RLM sub-calls.

2. **Walk forward (who cites the seed).** Run:
   ```bash
   python .claude/skills/citation-graph-research/scripts/citation_graph.py <seed> \
       --keywords <topic terms> --top 40
   ```
   Seeds can be an arXiv ID or URL, a DOI, or a title. Use `--only-matching` on
   heavily cited seeds, `--json` to save results, and `--direction references` or
   `both` to walk backward too.

3. **Triage.** Classify each relevant hit as *builds on*, *critiques or
   reproduces*, *competes with* or *mentions*. Use the "influential" flag and the
   quoted citation contexts. Read abstracts for the top ~10, and fetch the full
   text only for papers that bear directly on the question.

4. **Snowball one hop.** Run step 2 on the 2–3 most relevant citing papers. For
   multiple seeds, papers that cite *both* seeds are the most on-topic results.
   Treat the overlap as the core of the research area.

5. **Walk backward when positioning a claim.** Before calling something novel,
   check the seed's references and the references of the closest competitors.

6. **Web search only for gaps.** Use it for things citation indexes miss: very
   recent preprints (indexes lag days to weeks), code repositories, blog posts
   and release notes. Also search the seed's own GitHub repo and the authors'
   pages. Label these results as coming from outside the citation graph.

## Reporting rules

- Tag every claim: **[read]** for primary text read, **[abstract]** for abstract
  or API metadata only, **[snippet]** for a search-engine summary, **[inferred]**
  for your own reasoning. Never present a snippet as a verified result.
- Give the retrieval date and source (Semantic Scholar or OpenAlex). Citation
  counts for papers under a year old are incomplete.
- Organise findings by relationship to the seed (follow-ups, critiques,
  competitors, gaps), not as a flat list.
- End with **open gaps**: what the graph shows nobody has done yet.

## Network access

The script needs outbound HTTPS to `api.semanticscholar.org` (primary) and
`api.openalex.org` (fallback). Full-text reading needs `arxiv.org`. If the
environment's network policy blocks them, the script says so. In that case, say
this to the user, fall back to web search with the same seed-centred structure,
and lower confidence labels accordingly. Set `SEMANTIC_SCHOLAR_API_KEY` for
higher rate limits if one is available.
