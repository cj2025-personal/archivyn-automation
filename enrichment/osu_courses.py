"""
OSU Course Catalog collector.
Scrapes the public OSU class search to find courses taught by a professor.
Uses content.osu.edu class search API.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import names_match_from_query, names_match_fuzzy

logger = logging.getLogger(__name__)

# OSU class search API
CLASS_SEARCH_URL = "https://content.osu.edu/v2/classes/search"


class OSUCoursesCollector(BaseCollector):
    """Collect course information from OSU's class search."""

    def __init__(self, **kwargs):
        # OSU class search: multiple requests per professor (one per term)
        kwargs.setdefault("rate_limit_delay", 2.0)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "osu_courses"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        courses = await self._search_courses(query)
        if not courses:
            return self._make_result(query, success=False, error="No courses found for this professor")

        # Deduplicate by course code + title
        seen = set()
        unique_courses = []
        for c in courses:
            key = f"{c.get('subject', '')}{c.get('catalog_number', '')}:{c.get('title', '')}"
            if key not in seen:
                seen.add(key)
                unique_courses.append(c)

        data = {
            "total_courses": len(unique_courses),
            "courses": unique_courses,
        }

        raw_text = self._to_text(data, query)
        return self._make_result(query, success=True, data=data, raw_text=raw_text)

    async def _search_courses(self, query: ProfessorQuery) -> List[Dict]:
        """Search OSU class search for courses by instructor name."""
        all_courses = []

        # Try current and recent terms
        terms = self._get_recent_terms()

        # Try multiple query formats: full name first, then "Last, First"
        search_queries = [
            query.name,
            f"{query.last_name}, {query.first_name}",
            query.last_name,
        ]

        found_any = False
        for term in terms:
            for search_q in search_queries:
                if found_any and search_q == query.last_name:
                    # Skip the broad last-name-only search if we already found results
                    continue

                params = {
                    "q": search_q,
                    "term": term,
                    "campus": "col",  # Columbus campus
                    "p": 1,
                    "subject": "",
                }

                try:
                    resp = await self._get_json(CLASS_SEARCH_URL, params=params)
                    if not resp or not resp.get("data", {}).get("courses"):
                        continue

                    for course in resp["data"]["courses"]:
                        # Check if this professor actually teaches it
                        sections = course.get("sections", [])
                        prof_teaches = False
                        section_info = []

                        for section in sections:
                            for meeting in section.get("meetings", []):
                                instructors = meeting.get("instructors", [])
                                for inst in instructors:
                                    inst_name = inst.get("displayName", "")
                                    # Try strict match first, then fuzzy
                                    matched = names_match_from_query(query, inst_name)
                                    if not matched:
                                        matched = names_match_fuzzy(
                                            query.first_name, query.last_name, inst_name
                                        )
                                    if matched:
                                        prof_teaches = True
                                        section_info.append({
                                            "section": section.get("section", ""),
                                            "class_number": section.get("classNumber", ""),
                                            "component": section.get("component", ""),
                                            "days": meeting.get("days", ""),
                                            "start_time": meeting.get("startTime", ""),
                                            "end_time": meeting.get("endTime", ""),
                                            "building": meeting.get("buildingDescription", ""),
                                            "room": meeting.get("room", ""),
                                            "instructor": inst.get("displayName", ""),
                                        })

                        if prof_teaches:
                            all_courses.append({
                                "subject": course.get("subject", ""),
                                "catalog_number": course.get("catalogNumber", ""),
                                "title": course.get("title", ""),
                                "description": course.get("description", ""),
                                "term": term,
                                "term_name": self._term_name(term),
                                "credit_hours": course.get("minUnits", ""),
                                "academic_career": course.get("academicCareer", ""),
                                "sections": section_info,
                            })
                            found_any = True

                except Exception as e:
                    logger.warning("[osu_courses] Error searching term %s with query '%s': %s", term, search_q, e)
                    continue

                # If we found results with this query format, skip broader queries for this term
                if found_any:
                    break

        return all_courses

    @staticmethod
    def _get_recent_terms() -> List[str]:
        """Generate term codes for recent semesters.
        OSU term format: YYYYT where T is: 2=Spring, 4=Summer, 8=Autumn
        """
        from datetime import datetime
        year = datetime.now().year
        terms = []
        for y in range(year, year - 3, -1):
            terms.extend([f"{y}8", f"{y}2", f"{y}4"])  # Autumn, Spring, Summer
        return terms

    @staticmethod
    def _term_name(term_code: str) -> str:
        """Convert term code to readable name."""
        if len(term_code) != 5:
            return term_code
        year = term_code[:4]
        season = {"2": "Spring", "4": "Summer", "8": "Autumn"}.get(term_code[4], "")
        return f"{season} {year}"

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = []
        lines.append(f"=== OSU Courses Taught: {query.name} ===")
        lines.append(f"Total unique courses found: {data['total_courses']}")
        lines.append("")

        for i, course in enumerate(data["courses"], 1):
            code = f"{course['subject']} {course['catalog_number']}"
            lines.append(f"\n{i}. {code}: {course['title']}")
            lines.append(f"   Term: {course.get('term_name', course.get('term', 'N/A'))}")
            if course.get("credit_hours"):
                lines.append(f"   Credit hours: {course['credit_hours']}")
            if course.get("description"):
                lines.append(f"   Description: {course['description'][:400]}")
            if course.get("sections"):
                for sec in course["sections"][:3]:
                    time_str = ""
                    if sec.get("days") and sec.get("start_time"):
                        time_str = f" ({sec['days']} {sec['start_time']}-{sec['end_time']})"
                    lines.append(f"   Section: {sec.get('component', '')} {sec.get('section', '')}{time_str}")

        return "\n".join(lines)
