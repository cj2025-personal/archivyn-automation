"""
OSU course listings: which courses a professor teaches, from Ohio State's
public class search (content.osu.edu/v2/classes/search).

Two corrections from the 2026 run, where it found nobody (0 of 460):
  - OSU term codes are four digits, "1" + two-digit year + season
    (1262 Spring 2026, 1264 Summer 2026, 1268 Autumn 2026); the collector
    sent "20268", which matches no term.
  - Each result nests the course under `course`; the collector read subject
    and title from the outer object.

What is kept is what a course is - code, title, catalog description, term,
credits, and the professor's role in it. Not when or where it meets: days,
times, buildings and rooms say where a named person will be each week, which
is no part of describing their teaching. Not instructors' emails either.
"""

import logging
from datetime import datetime
from typing import Dict, List

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import names_match_fuzzy, names_match_from_query

logger = logging.getLogger(__name__)

CLASS_SEARCH_URL = "https://content.osu.edu/v2/classes/search"
SEASONS = {"2": "Spring", "4": "Summer", "8": "Autumn"}


def recent_terms(now: datetime = None, years: int = 3) -> List[str]:
    """OSU term codes, newest first: this year and the two before it."""
    year = (now or datetime.now()).year
    return [f"1{y % 100:02d}{s}" for y in range(year, year - years, -1) for s in ("8", "4", "2")]


def term_name(code: str) -> str:
    if len(code) != 4 or not code.isdigit():
        return code
    return f"{SEASONS.get(code[3], '')} 20{code[1:3]}".strip()


def teaching_role(query: ProfessorQuery, sections: List[Dict]) -> str:
    """The professor's role in a course's sections ('' if they don't teach it)."""
    for section in sections or []:
        for meeting in section.get("meetings") or []:
            for inst in meeting.get("instructors") or []:
                name = inst.get("displayName") or ""
                if names_match_from_query(query, name) or names_match_fuzzy(query.first_name, query.last_name, name):
                    return inst.get("role") or "instructor"
    return ""


def course_record(query: ProfessorQuery, result: Dict, term: str) -> Dict:
    """One search result as a course the professor teaches, or {} if they don't."""
    course = result.get("course") or {}
    role = teaching_role(query, result.get("sections") or [])
    if not role or not course:
        return {}
    return {
        "subject": course.get("subject", ""),
        "catalog_number": course.get("catalogNumber", ""),
        "title": course.get("title", ""),
        "description": (course.get("description") or "")[:600],
        "term": term,
        "term_name": term_name(term),
        "credit_hours": course.get("minUnits", ""),
        "academic_career": course.get("academicCareer", ""),
        "role": role,
    }


class OSUCoursesCollector(BaseCollector):
    """Courses a professor teaches, from OSU's class search."""

    def __init__(self, **kwargs):
        # One request per term and query form; be gentle with the university.
        kwargs.setdefault("rate_limit_delay", 2.0)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "osu_courses"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        courses: List[Dict] = []
        seen = set()
        for term in recent_terms():
            for q in (query.name, query.last_name):
                if not q or len(q) < 2:
                    continue
                resp = await self._get_json(CLASS_SEARCH_URL, params={"q": q, "term": term, "campus": "col", "p": 1})
                found = False
                for result in ((resp or {}).get("data") or {}).get("courses") or []:
                    record = course_record(query, result, term)
                    if not record:
                        continue
                    found = True
                    key = (record["subject"], record["catalog_number"], record["title"])
                    if key not in seen:
                        seen.add(key)
                        courses.append(record)
                if found:
                    break  # the full name matched this term; no need for the broader query
        if not courses:
            return self._make_result(query, success=False, error="No courses found for this professor")
        data = {"total_courses": len(courses), "courses": courses}
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = [f"=== OSU Courses Taught: {query.name} ===", f"Total unique courses found: {data['total_courses']}", ""]
        for i, course in enumerate(data["courses"], 1):
            lines.append(f"\n{i}. {course['subject']} {course['catalog_number']}: {course['title']}")
            lines.append(f"   Term: {course.get('term_name') or course.get('term')}")
            if course.get("credit_hours"):
                lines.append(f"   Credit hours: {course['credit_hours']}")
            if course.get("description"):
                lines.append(f"   Description: {course['description'][:400]}")
        return "\n".join(lines)
