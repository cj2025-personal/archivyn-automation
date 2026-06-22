"""
Inline fixture-based tests for collect_osu_scholar_urls helpers.
No live DDG / Mongo. Run with: python _test_collect_logic.py
"""
import sys
sys.path.insert(0, '.')

from collect_osu_scholar_urls import (
    _identity_match, _should_skip, _identity_keywords,
    _build_queries, _url_priority, _normalize_url, _slugify,
)


def banner(t):
    print(f"\n=== {t} ===")


# Mongo-shaped fixture scholars
SCHOLARS = [
    {
        'name': {'full': 'Rama Yedavalli', 'first': 'Rama', 'last': 'Yedavalli'},
        'metadata': {'field_of_study': 'Dynamics and Control Systems'},
        'about': {'department': 'Department of Mechanical and Aerospace Engineering'},
    },
    {
        'name': {'full': 'Kevin Brown', 'first': 'Kevin', 'last': 'Brown'},
        'metadata': {'field_of_study': 'Law'},
        'about': {'department': 'Maurer School of Law'},
    },
    {
        'name': {'full': 'Max Tu', 'first': 'Max', 'last': 'Tu'},
        'metadata': {'field_of_study': 'Marketing'},
        'about': {'department': 'Fisher College of Business'},
    },
    {
        'name': {'full': 'Ashley Lipps', 'first': 'Ashley', 'last': 'Lipps'},
        'metadata': {'field_of_study': 'Medicine'},
        'about': {'department': ''},
    },
]


banner("Test 1: identity match — true positives (should KEEP)")
positives = [
    (SCHOLARS[0], {'url': 'https://mae.osu.edu/people/yedavalli.1',
                   'title': 'Rama Yedavalli | Mechanical and Aerospace Engineering',
                   'snippet': 'Professor at Ohio State University specializing in dynamics and control systems.'}),
    (SCHOLARS[0], {'url': 'https://scholar.google.com/citations?user=abc',
                   'title': 'Rama Yedavalli - Google Scholar',
                   'snippet': 'Ohio State University. Verified email at osu.edu. Control systems, robust control.'}),
    (SCHOLARS[1], {'url': 'https://moritzlaw.osu.edu/about/people/kevin-brown',
                   'title': 'Kevin Brown - Moritz College of Law',
                   'snippet': 'Kevin Brown is a Professor at the Moritz College of Law at Ohio State.'}),
    (SCHOLARS[2], {'url': 'https://fisher.osu.edu/people/tu.123',
                   'title': 'Max Tu | Fisher College of Business',
                   'snippet': 'Marketing professor at Ohio State University.'}),
    (SCHOLARS[3], {'url': 'https://medicine.osu.edu/people/ashley-lipps',
                   'title': 'Ashley Lipps, MD',
                   'snippet': 'Ashley Lipps is a physician at the Ohio State Wexner Medical Center.'}),
]
for s, r in positives:
    m = _identity_match(r, _identity_keywords(s))
    print(f"  {('PASS' if m else 'FAIL'):4s}  {s['name']['full']:18s} <- {r['url'][:60]}")


banner("Test 2: identity match — true negatives (should DROP)")
negatives = [
    (SCHOLARS[1], {'url': 'https://example.com/blog/12345',
                   'title': 'Kevin Brown signs new record deal',
                   'snippet': 'Pop star Kevin Brown announces world tour dates.'}),
    (SCHOLARS[1], {'url': 'https://random-blog.net/post',
                   'title': 'Famous Kevin Browns through history',
                   'snippet': 'A look at five notable people named Kevin Brown.'}),
    (SCHOLARS[0], {'url': 'https://example.com/article',
                   'title': 'Different person Yedavalli',
                   'snippet': 'Some unrelated content about a chef.'}),
    (SCHOLARS[3], {'url': 'https://example.com/news',
                   'title': 'Ashley Lipps wins photography award',
                   'snippet': 'Photographer Ashley Lipps from Cleveland wins national prize.'}),
]
for s, r in negatives:
    m = _identity_match(r, _identity_keywords(s))
    print(f"  {('PASS' if not m else 'FAIL'):4s}  {s['name']['full']:18s} <- {r['url'][:60]}")


banner("Test 3: skip-domain filter")
skip_cases = [
    ('https://www.linkedin.com/in/rama-yedavalli', True),
    ('https://twitter.com/somebody', True),
    ('https://www.facebook.com/profile', True),
    ('https://help.pbs.org/support/12345', True),
    ('https://www.osu.edu/faculty/yedavalli', False),
    ('https://scholar.google.com/citations?user=xx', False),
    ('https://orcid.org/0000-0001-2345-6789', False),
]
for url, expected_skip in skip_cases:
    actual = _should_skip(url)
    print(f"  {('PASS' if actual==expected_skip else 'FAIL'):4s}  expect_skip={expected_skip}  actual={actual}  {url[:55]}")


banner("Test 4: URL priority (lower = higher rank)")
urls_sorted = sorted([
    'https://www.osu.edu/faculty/yedavalli',
    'https://scholar.google.com/citations?user=x',
    'https://orcid.org/abc',
    'https://www.researchgate.net/profile/x',
    'https://www.somerandomblog.com/post',
    'https://arxiv.org/abs/1234',
    'https://www.someuniversity.edu/people/y',
    'https://www.springer.com/article/x',
], key=_url_priority)
for u in urls_sorted:
    print(f"  prio={_url_priority(u):3d}  {u}")


banner("Test 5: query construction")
qs = _build_queries(_identity_keywords(SCHOLARS[0]))
print(f"  generated {len(qs)} queries (sample 6):")
for q in qs[:6]:
    print(f"    {q[:90]}")


banner("Test 6: URL normalization")
norm_cases = [
    ('https://www.osu.edu/faculty/x?utm_source=google&utm_medium=cpc', 'https://osu.edu/faculty/x'),
    ('https://scholar.google.com/citations?user=ABC123&hl=en', 'https://scholar.google.com/citations?user=ABC123&hl=en'),
    ('http://example.com/page/', 'http://example.com/page'),
    ('https://example.com', 'https://example.com/'),
]
for raw, want in norm_cases:
    got = _normalize_url(raw)
    print(f"  {('PASS' if got==want else 'FAIL'):4s}  {raw[:55]}  ->  {got}")


banner("Test 7: slug")
slug_cases = [
    ('Rama Yedavalli', 'abcd1234-5678-9012-3456-789012345678', 'rama-yedavalli-abcd1234'),
    ('Dr. Shirley Ann Jackson', 'ff17a5ba-b46a-4157-bee7-ede7aafaefd1', 'dr-shirley-ann-jackson-ff17a5ba'),
    ('Max Tu', 'short-id', 'max-tu-short-id'),
]
for name, pid, want in slug_cases:
    got = _slugify(name, pid)
    print(f"  {('PASS' if got==want else 'FAIL'):4s}  {name!r:30s} -> {got!r}")


banner("Test 8: regression cases — previously over-rejected")
# These used to get filtered out because the snippet didn't say
# "Ohio State" — now they should KEEP because the URL is osu.edu
# (or has the name in the slug).
regression_keep = [
    # OSU host, name in slug, no "Ohio State" in snippet
    (SCHOLARS[0], {'url': 'https://mae.osu.edu/people/yedavalli.1',
                   'title': 'Yedavalli – Mechanical and Aerospace Engineering',
                   'snippet': 'Faculty profile, research interests robust control.'}),
    # OSU subdomain + first+last in URL slug, snippet useless
    (SCHOLARS[1], {'url': 'https://moritzlaw.osu.edu/about/people/kevin-brown',
                   'title': 'Kevin Brown',
                   'snippet': 'Faculty page.'}),
    # Common short last name "Tu" — name in URL path
    (SCHOLARS[2], {'url': 'https://fisher.osu.edu/people/max.tu.1',
                   'title': 'Max Tu',
                   'snippet': 'Marketing.'}),
    # Generic .edu page, name match in snippet, no OSU
    ({'name': {'full': 'Andre Palmer', 'first': 'Andre', 'last': 'Palmer'},
      'metadata': {'field_of_study': 'Chemical Engineering'},
      'about': {'department': 'Chemical and Biomolecular Engineering'}},
     {'url': 'https://www.someuniversity.edu/news/2021-symposium',
      'title': 'Andre Palmer keynote at 2021 Symposium',
      'snippet': 'Andre Palmer presented research on hemoglobin oxygen transfer.'}),
]
for s, r in regression_keep:
    m = _identity_match(r, _identity_keywords(s))
    print(f"  {('PASS' if m else 'FAIL'):4s}  {s['name']['full']:18s} (KEEP) <- {r['url'][:60]}")


banner("Test 9: short-last-name false-positive guards")
# "Tu" (short) should NOT match a random URL containing "tu" (e.g. tutorial).
# "Yedavalli" (distinctive) IS allowed to match on last name alone.
short_name_cases = [
    # Random URL with "tu" substring → DROP
    (SCHOLARS[2], {'url': 'https://example.com/tutorials/marketing-101',
                   'title': 'Marketing tutorials',
                   'snippet': 'Learn marketing fundamentals.'}, False),
    # "Tu" in path but NO first name and NO OSU/edu → DROP
    (SCHOLARS[2], {'url': 'https://random.com/tu-cafe',
                   'title': 'Tu Cafe',
                   'snippet': 'A nice café.'}, False),
    # Yedavalli alone in title (distinctive last name) on .edu → KEEP
    (SCHOLARS[0], {'url': 'https://other.edu/news/yedavalli-talk-2023',
                   'title': 'Yedavalli talk at 2023 ASME conference',
                   'snippet': 'Robust control session.'}, True),
]
for s, r, expected_keep in short_name_cases:
    m = _identity_match(r, _identity_keywords(s))
    ok = m == expected_keep
    print(f"  {('PASS' if ok else 'FAIL'):4s}  expect_keep={expected_keep} actual={m}  {s['name']['full']:18s} <- {r['url'][:55]}")
