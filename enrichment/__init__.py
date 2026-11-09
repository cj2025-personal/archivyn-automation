"""
Enrichment Pipeline — collects public data from multiple sources
to enrich existing professor profiles.

Sources:
- Web Search (Brave Search API — finds actual pages about each professor)
- Semantic Scholar (publications, citations; identified through OpenAlex DOIs)
- OpenAlex (publications, concepts, institutions)
- CrossRef (DOI metadata, funding info)
- ORCID (career history, education, employment)
- NSF Award Search (federal grants)
- NIH RePORTER (NIH-funded grants)
- OSU Course Catalog (courses taught)
- YouTube (lectures, talks)
- OSU News (press releases, media mentions)
- Google News (media coverage)
- OSU Knowledge Bank (works authored and theses advised)

Every page fetch follows crawl_policy: one honest name, robots.txt
respected, and a refusal is final. Google Scholar, RateMyProfessors and
YouTube transcripts are not collected: their terms forbid automated access.
"""
