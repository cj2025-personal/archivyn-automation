"""
RateMyProfessors collector.
Scrapes public professor ratings from RateMyProfessors.com using their
GraphQL API endpoint (same as their frontend uses).

Returns: overall rating, difficulty, "would take again" %, top tags, reviews.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery
from .validation import names_match

logger = logging.getLogger(__name__)

# RMP uses a GraphQL endpoint
RMP_GRAPHQL_URL = "https://www.ratemyprofessors.com/graphql"

# Ohio State University school ID on RMP
OSU_SCHOOL_ID = "U2Nob29sLTcyNA=="  # base64 encoded


class RateMyProfessorCollector(BaseCollector):
    """Collect student ratings and reviews from RateMyProfessors."""

    def __init__(self, **kwargs):
        # RateMyProfessors GraphQL: aggressive rate limiting
        kwargs.setdefault("rate_limit_delay", 4.0)
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "rate_my_professor"

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        # Step 1: Search for the professor at OSU
        teacher = await self._search_teacher(query)
        if not teacher:
            return self._make_result(query, success=False, error="Professor not found on RateMyProfessors")

        teacher_id = teacher.get("id", "")

        # Step 2: Get detailed ratings + reviews
        detail = await self._get_teacher_detail(teacher_id)
        if detail:
            teacher.update(detail)

        data = {
            "rmp_id": teacher_id,
            "name": f"{teacher.get('firstName', '')} {teacher.get('lastName', '')}".strip(),
            "department": teacher.get("department", ""),
            "overall_rating": teacher.get("avgRating", None),
            "difficulty_rating": teacher.get("avgDifficulty", None),
            "would_take_again_pct": teacher.get("wouldTakeAgainPercent", None),
            "num_ratings": teacher.get("numRatings", 0),
            "top_tags": teacher.get("teacherRatingTags", []),
            "courses_taught": teacher.get("courseCodes", []),
            "reviews": teacher.get("reviews", []),
        }

        raw_text = self._to_text(data, query)
        return self._make_result(query, success=True, data=data, raw_text=raw_text)

    async def _search_teacher(self, query: ProfessorQuery) -> Optional[Dict]:
        """Search RMP for a teacher at OSU."""
        graphql_query = """
        query TeacherSearchQuery($query: TeacherSearchQuery!) {
            newSearch {
                teachers(query: $query) {
                    edges {
                        node {
                            id
                            firstName
                            lastName
                            school {
                                id
                                name
                            }
                            department
                            avgRating
                            avgDifficulty
                            wouldTakeAgainPercent
                            numRatings
                        }
                    }
                }
            }
        }
        """
        variables = {
            "query": {
                "text": query.name,
                "schoolID": OSU_SCHOOL_ID,
            }
        }

        try:
            client = await self.get_client()
            await self._rate_limit()
            resp = await client.post(
                RMP_GRAPHQL_URL,
                json={"query": graphql_query, "variables": variables},
                headers={
                    "Authorization": "Basic dGVzdDp0ZXN0",  # RMP's public auth token
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("[rate_my_professor] Search failed for %s: %s", query.name, e)
            return None

        edges = (
            data.get("data", {})
            .get("newSearch", {})
            .get("teachers", {})
            .get("edges", [])
        )

        if not edges:
            return None

        # Strict match: require both first name + last name
        for edge in edges:
            node = edge.get("node", {})
            found_name = f"{node.get('firstName', '')} {node.get('lastName', '')}".strip()
            if names_match(query.first_name, query.last_name, found_name):
                return node

        # No match found — do NOT fall back to first result (would return wrong person)
        return None

    async def _get_teacher_detail(self, teacher_id: str) -> Optional[Dict]:
        """Get detailed teacher info including reviews and tags."""
        graphql_query = """
        query TeacherRatingsQuery($id: ID!) {
            node(id: $id) {
                ... on Teacher {
                    id
                    firstName
                    lastName
                    department
                    avgRating
                    avgDifficulty
                    wouldTakeAgainPercent
                    numRatings
                    teacherRatingTags {
                        tagName
                        tagCount
                    }
                    courseCodes {
                        courseName
                        courseCount
                    }
                    ratings(first: 20) {
                        edges {
                            node {
                                comment
                                qualityRating
                                difficultyRating
                                class
                                date
                                isForOnlineClass
                                wouldTakeAgain
                                grade
                                thumbsUpTotal
                                thumbsDownTotal
                                attendanceMandatory
                                textbookUse
                            }
                        }
                    }
                }
            }
        }
        """
        variables = {"id": teacher_id}

        try:
            client = await self.get_client()
            await self._rate_limit()
            resp = await client.post(
                RMP_GRAPHQL_URL,
                json={"query": graphql_query, "variables": variables},
                headers={
                    "Authorization": "Basic dGVzdDp0ZXN0",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("[rate_my_professor] Detail fetch failed for %s: %s", teacher_id, e)
            return None

        node = data.get("data", {}).get("node", {})
        if not node:
            return None

        # Extract reviews
        reviews = []
        for edge in (node.get("ratings", {}).get("edges", []))[:20]:
            r = edge.get("node", {})
            reviews.append({
                "comment": r.get("comment", ""),
                "quality": r.get("qualityRating"),
                "difficulty": r.get("difficultyRating"),
                "class": r.get("class", ""),
                "date": r.get("date", ""),
                "would_take_again": r.get("wouldTakeAgain"),
                "grade": r.get("grade", ""),
                "online": r.get("isForOnlineClass", False),
                "thumbs_up": r.get("thumbsUpTotal", 0),
                "thumbs_down": r.get("thumbsDownTotal", 0),
            })

        # Extract tags
        tags = [
            {"name": t.get("tagName", ""), "count": t.get("tagCount", 0)}
            for t in (node.get("teacherRatingTags") or [])
        ]
        tags.sort(key=lambda t: t["count"], reverse=True)

        # Extract courses
        courses = [
            {"name": c.get("courseName", ""), "count": c.get("courseCount", 0)}
            for c in (node.get("courseCodes") or [])
        ]

        return {
            "teacherRatingTags": tags,
            "courseCodes": courses,
            "reviews": reviews,
        }

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = []
        lines.append(f"=== RateMyProfessors: {data['name']} ===")
        lines.append(f"Department: {data.get('department', 'N/A')}")
        if data.get("overall_rating") is not None:
            lines.append(f"Overall rating: {data['overall_rating']}/5.0")
        if data.get("difficulty_rating") is not None:
            lines.append(f"Difficulty: {data['difficulty_rating']}/5.0")
        if data.get("would_take_again_pct") is not None and data["would_take_again_pct"] >= 0:
            lines.append(f"Would take again: {data['would_take_again_pct']:.0f}%")
        lines.append(f"Number of ratings: {data.get('num_ratings', 0)}")

        if data.get("top_tags"):
            lines.append("\n--- Student Tags ---")
            for tag in data["top_tags"][:10]:
                lines.append(f"  - {tag['name']} ({tag['count']} mentions)")

        if data.get("courses_taught"):
            lines.append("\n--- Courses Rated ---")
            for course in data["courses_taught"][:15]:
                lines.append(f"  - {course['name']} ({course['count']} ratings)")

        if data.get("reviews"):
            lines.append("\n--- Student Reviews ---")
            for i, rev in enumerate(data["reviews"][:15], 1):
                lines.append(f"\nReview {i}: Quality {rev.get('quality', 'N/A')}/5, Difficulty {rev.get('difficulty', 'N/A')}/5")
                if rev.get("class"):
                    lines.append(f"  Course: {rev['class']}")
                if rev.get("grade"):
                    lines.append(f"  Grade received: {rev['grade']}")
                if rev.get("comment"):
                    lines.append(f"  \"{rev['comment'][:300]}\"")

        return "\n".join(lines)
