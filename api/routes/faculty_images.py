"""
Faculty image viewer routes.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Dict, List

import boto3
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name


DEFAULT_UNIVERSITY = "Ohio State University-Main Campus"

router = APIRouter(tags=["faculty-images"])


class FacultyImageRow(BaseModel):
    profile_id: str
    name: str
    university: str
    image_url: str
    processed_status: str


class FacultyImageListResponse(BaseModel):
    page: int
    page_size: int
    total_matches: int
    rows: List[FacultyImageRow]


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable: {name}")


@lru_cache(maxsize=1)
def get_faculty_images_collection():
    mongodb_uri = _require_env("MONGO_ATLAS_URI")
    database_name = resolve_mongo_db_name(mongodb_uri, default="FacultyImages")
    collection_name = os.getenv("MONGODB_COLLECTION_NAME", "images")
    client = create_mongo_client(mongodb_uri)
    return client[database_name][collection_name]


@lru_cache(maxsize=1)
def get_scholars_collection():
    mongodb_uri = _require_env("MONGODB_URI")
    database_name = resolve_mongo_db_name(mongodb_uri)
    client = create_mongo_client(mongodb_uri)
    return client[database_name]["scholars"]


@lru_cache(maxsize=1)
def get_s3_client():
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=_require_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=_require_env("AWS_SECRET_ACCESS_KEY"),
    )


def build_query(university: str, search: str):
    query = {
        "YOLOv8n_human_detection.has_human": True,
        "image.status": "uploaded",
    }

    if university:
        query["university"] = {"$regex": re.escape(university), "$options": "i"}

    if search:
        escaped_search = re.escape(search)
        query["$or"] = [
            {"name": {"$regex": escaped_search, "$options": "i"}},
            {"university": {"$regex": escaped_search, "$options": "i"}},
        ]

    return query


def build_s3_uri(bucket_name: str, object_key: str) -> str:
    return f"s3://{bucket_name}/{object_key}"


def load_processed_statuses(profile_ids: List[str], s3_uris: List[str]) -> Dict[str, str]:
    statuses = {profile_id: "pending" for profile_id in profile_ids}
    if not profile_ids and not s3_uris:
        return statuses

    scholars_collection = get_scholars_collection()
    scholar_query = {
        "$or": [
            {"image_mapping.image_profile_id": {"$in": profile_ids}},
            {"image_mapping.s3_uri": {"$in": s3_uris}},
            {"about.avatar_url": {"$in": s3_uris}},
            {"display.profile_image_url": {"$in": s3_uris}},
        ]
    }
    projection = {
        "_id": 0,
        "image_mapping.image_profile_id": 1,
        "image_mapping.s3_uri": 1,
        "about.avatar_url": 1,
        "display.profile_image_url": 1,
    }

    done_profile_ids = set()
    done_s3_uris = set()
    for scholar_doc in scholars_collection.find(scholar_query, projection):
        image_mapping = scholar_doc.get("image_mapping") or {}
        about = scholar_doc.get("about") or {}
        display = scholar_doc.get("display") or {}

        image_profile_id = str(image_mapping.get("image_profile_id") or "").strip()
        if image_profile_id:
            done_profile_ids.add(image_profile_id)

        for uri in (
            str(image_mapping.get("s3_uri") or "").strip(),
            str(about.get("avatar_url") or "").strip(),
            str(display.get("profile_image_url") or "").strip(),
        ):
            if uri:
                done_s3_uris.add(uri)

    for profile_id, s3_uri in zip(profile_ids, s3_uris):
        if profile_id in done_profile_ids or s3_uri in done_s3_uris:
            statuses[profile_id] = "done"

    return statuses


@router.get("/api/faculty-images", response_model=FacultyImageListResponse)
async def list_faculty_images(
    request: Request,
    university: str = Query(
        DEFAULT_UNIVERSITY,
        description="Case-insensitive university filter",
    ),
    search: str = Query("", description="Case-insensitive name/university search"),
    processed_status: str = Query(
        "all",
        description="Filter by processing status: all, done, pending",
    ),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    try:
        collection = get_faculty_images_collection()
        base_query = build_query(university=university.strip(), search=search.strip())
        normalized_processed_status = processed_status.strip().lower() or "all"
        if normalized_processed_status not in {"all", "done", "pending"}:
            raise HTTPException(
                status_code=400,
                detail="processed_status must be one of: all, done, pending",
            )

        skip = (page - 1) * page_size

        projection = {
            "_id": 0,
            "profile_id": 1,
            "name": 1,
            "university": 1,
            "image.s3_bucket": 1,
            "image.s3_key": 1,
        }
        if normalized_processed_status == "all":
            total_matches = collection.count_documents(base_query)
            cursor = (
                collection.find(base_query, projection)
                .sort([("name", 1), ("profile_id", 1)])
                .skip(skip)
                .limit(page_size)
            )

            page_documents = []
            profile_ids: List[str] = []
            s3_uris: List[str] = []
            for doc in cursor:
                profile_id = str(doc.get("profile_id", "")).strip()
                if not profile_id:
                    continue
                image_info = doc.get("image") or {}
                bucket_name = str(image_info.get("s3_bucket") or os.getenv("BUCKET_NAME") or "").strip()
                object_key = str(image_info.get("s3_key") or "").strip()
                s3_uri = build_s3_uri(bucket_name, object_key) if bucket_name and object_key else ""
                profile_ids.append(profile_id)
                s3_uris.append(s3_uri)
                page_documents.append((doc, profile_id))

            processed_statuses = load_processed_statuses(profile_ids, s3_uris)
        else:
            all_matching_documents = []
            all_profile_ids: List[str] = []
            all_s3_uris: List[str] = []
            cursor = collection.find(base_query, projection).sort([("name", 1), ("profile_id", 1)])
            for doc in cursor:
                profile_id = str(doc.get("profile_id", "")).strip()
                if not profile_id:
                    continue
                image_info = doc.get("image") or {}
                bucket_name = str(image_info.get("s3_bucket") or os.getenv("BUCKET_NAME") or "").strip()
                object_key = str(image_info.get("s3_key") or "").strip()
                s3_uri = build_s3_uri(bucket_name, object_key) if bucket_name and object_key else ""
                all_profile_ids.append(profile_id)
                all_s3_uris.append(s3_uri)
                all_matching_documents.append((doc, profile_id, s3_uri))

            processed_statuses = load_processed_statuses(all_profile_ids, all_s3_uris)
            filtered_documents = [
                (doc, profile_id)
                for doc, profile_id, _s3_uri in all_matching_documents
                if processed_statuses.get(profile_id, "pending") == normalized_processed_status
            ]
            total_matches = len(filtered_documents)
            page_documents = filtered_documents[skip : skip + page_size]

        rows = []
        for doc, profile_id in page_documents:
            rows.append(
                FacultyImageRow(
                    profile_id=profile_id,
                    name=(doc.get("name") or "Unknown").strip(),
                    university=(doc.get("university") or "Unknown").strip(),
                    image_url=str(request.url_for("faculty_image_content", profile_id=profile_id)),
                    processed_status=processed_statuses.get(profile_id, "pending"),
                )
            )

        return FacultyImageListResponse(
            page=page,
            page_size=page_size,
            total_matches=total_matches,
            rows=rows,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load faculty images: {exc}",
        ) from exc


@router.get("/api/faculty-images/content/{profile_id}", name="faculty_image_content")
async def faculty_image_content(profile_id: str):
    try:
        collection = get_faculty_images_collection()
        document = collection.find_one(
            {
                "profile_id": profile_id,
                "YOLOv8n_human_detection.has_human": True,
                "image.status": "uploaded",
            },
            {
                "_id": 0,
                "image.s3_bucket": 1,
                "image.s3_key": 1,
                "image.content_type": 1,
            },
        )
        if not document:
            raise HTTPException(status_code=404, detail="Faculty image not found")

        image_info = document.get("image") or {}
        bucket_name = image_info.get("s3_bucket") or os.getenv("BUCKET_NAME")
        object_key = image_info.get("s3_key")
        content_type = image_info.get("content_type") or "application/octet-stream"

        if not bucket_name or not object_key:
            raise HTTPException(status_code=500, detail="Faculty image is missing S3 metadata")

        s3_object = get_s3_client().get_object(Bucket=bucket_name, Key=object_key)
        body = s3_object["Body"].read()

        return Response(
            content=body,
            media_type=content_type,
            headers={"Cache-Control": "public, max-age=3600"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch faculty image: {exc}",
        ) from exc


@router.get("/faculty-images", response_class=HTMLResponse)
async def faculty_images_view():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Faculty Images</title>
    <style>
        :root {
            --bg: #f6f2e8;
            --panel: rgba(255, 251, 243, 0.92);
            --ink: #1f2b24;
            --muted: #6b766b;
            --line: #d8d0c0;
            --accent: #0f5c4a;
            --accent-soft: #dcefe8;
            --accent-strong: #08372d;
            --shadow: 0 22px 48px rgba(31, 43, 36, 0.14);
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            color: var(--ink);
            font-family: "Trebuchet MS", "Gill Sans", "Segoe UI", sans-serif;
            background:
                radial-gradient(circle at top left, rgba(15, 92, 74, 0.18), transparent 28%),
                radial-gradient(circle at bottom right, rgba(190, 144, 56, 0.18), transparent 24%),
                linear-gradient(180deg, #efe7d7 0%, var(--bg) 46%, #f9f5ec 100%);
            padding: 32px 20px 48px;
        }

        .shell {
            max-width: 1400px;
            margin: 0 auto;
        }

        .hero {
            background: linear-gradient(135deg, rgba(255, 251, 243, 0.96), rgba(244, 237, 221, 0.92));
            border: 1px solid rgba(216, 208, 192, 0.8);
            border-radius: 24px;
            box-shadow: var(--shadow);
            padding: 28px;
            margin-bottom: 20px;
        }

        .eyebrow {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent-strong);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        h1 {
            margin: 16px 0 8px;
            font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
            font-size: clamp(2rem, 4vw, 3.5rem);
            line-height: 1;
        }

        .hero p {
            margin: 0;
            color: var(--muted);
            font-size: 1rem;
            max-width: 760px;
        }

        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 14px;
            margin-top: 22px;
        }

        .field {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .field label {
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: var(--muted);
        }

        .field input {
            width: 100%;
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 13px 14px;
            background: rgba(255, 255, 255, 0.92);
            color: var(--ink);
            font-size: 1rem;
        }

        .field select {
            width: 100%;
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 13px 14px;
            background: rgba(255, 255, 255, 0.92);
            color: var(--ink);
            font-size: 1rem;
        }

        .field input:focus {
            outline: 2px solid rgba(15, 92, 74, 0.16);
            border-color: var(--accent);
        }

        .field select:focus {
            outline: 2px solid rgba(15, 92, 74, 0.16);
            border-color: var(--accent);
        }

        .actions {
            display: flex;
            align-items: end;
            gap: 10px;
            flex-wrap: wrap;
        }

        button {
            border: 0;
            border-radius: 14px;
            padding: 13px 18px;
            cursor: pointer;
            font-size: 0.98rem;
            font-weight: 700;
            transition: transform 0.18s ease, box-shadow 0.18s ease, opacity 0.18s ease;
        }

        button:hover:not(:disabled) {
            transform: translateY(-1px);
            box-shadow: 0 10px 20px rgba(15, 92, 74, 0.14);
        }

        button:disabled {
            opacity: 0.45;
            cursor: not-allowed;
        }

        .primary {
            background: var(--accent);
            color: #f8faf8;
        }

        .secondary {
            background: #ebe4d5;
            color: var(--ink);
        }

        .meta {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
            margin: 18px 2px 0;
            color: var(--muted);
            font-size: 0.95rem;
        }

        .table-wrap {
            background: var(--panel);
            border: 1px solid rgba(216, 208, 192, 0.8);
            border-radius: 24px;
            box-shadow: var(--shadow);
            overflow: hidden;
        }

        .table-scroll {
            overflow: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            min-width: 920px;
        }

        thead th {
            position: sticky;
            top: 0;
            z-index: 1;
            text-align: left;
            padding: 16px 18px;
            background: rgba(234, 244, 239, 0.96);
            border-bottom: 1px solid var(--line);
            font-size: 0.8rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--accent-strong);
        }

        tbody td {
            padding: 18px;
            border-bottom: 1px solid rgba(216, 208, 192, 0.7);
            vertical-align: top;
            background: rgba(255, 251, 243, 0.7);
        }

        tbody tr:nth-child(even) td {
            background: rgba(250, 246, 238, 0.95);
        }

        .name-cell strong {
            display: block;
            margin-bottom: 4px;
            font-size: 1rem;
        }

        .subtle {
            color: var(--muted);
            font-size: 0.92rem;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 84px;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .pill.done {
            background: rgba(15, 92, 74, 0.12);
            color: var(--accent-strong);
            border: 1px solid rgba(15, 92, 74, 0.2);
        }

        .pill.pending {
            background: rgba(190, 144, 56, 0.12);
            color: #7a5505;
            border: 1px solid rgba(190, 144, 56, 0.22);
        }

        .image-link {
            color: var(--accent);
            font-weight: 700;
            text-decoration: none;
            word-break: break-all;
        }

        .image-link:hover {
            text-decoration: underline;
        }

        .thumb {
            width: 96px;
            height: 96px;
            object-fit: cover;
            border-radius: 18px;
            border: 1px solid rgba(15, 92, 74, 0.14);
            box-shadow: 0 10px 18px rgba(31, 43, 36, 0.1);
            background: #efe9dc;
        }

        .status,
        .empty {
            padding: 28px;
            color: var(--muted);
        }

        .status {
            display: none;
        }

        @media (max-width: 800px) {
            body {
                padding: 18px 12px 28px;
            }

            .hero,
            .table-wrap {
                border-radius: 18px;
            }

            .actions {
                width: 100%;
            }

            .actions button {
                flex: 1 1 0;
            }
        }
    </style>
</head>
<body>
    <div class="shell">
        <section class="hero">
            <span class="eyebrow">Faculty Image Archive</span>
            <h1>Ohio State faculty images</h1>
            <p>
                This view lists only image records where <code>YOLOv8n_human_detection.has_human=true</code>
                and <code>image.status=uploaded</code>. Each image is served through this app from the S3 path
                stored in Mongo.
            </p>
            <div class="controls">
                <div class="field">
                    <label for="universityInput">University</label>
                    <input id="universityInput" type="text" value="Ohio State University-Main Campus">
                </div>
                <div class="field">
                    <label for="searchInput">Search name</label>
                    <input id="searchInput" type="text" placeholder="Aaron, chemistry, engineering">
                </div>
                <div class="field">
                    <label for="pageSizeInput">Rows per page</label>
                    <input id="pageSizeInput" type="number" min="1" max="200" value="50">
                </div>
                <div class="field">
                    <label for="statusFilter">Processed</label>
                    <select id="statusFilter">
                        <option value="all">All</option>
                        <option value="done">Done</option>
                        <option value="pending">Unprocessed</option>
                    </select>
                </div>
                <div class="actions">
                    <button class="primary" id="loadButton" type="button">Load table</button>
                    <button class="secondary" id="prevButton" type="button">Previous</button>
                    <button class="secondary" id="nextButton" type="button">Next</button>
                </div>
            </div>
            <div class="meta">
                <div id="summary">Ready.</div>
                <div id="pageIndicator">Page 1</div>
            </div>
        </section>

        <section class="table-wrap">
            <div class="status" id="loadingState">Loading faculty image records...</div>
            <div class="table-scroll">
                <table>
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Status</th>
                            <th>University</th>
                            <th>Image URL</th>
                            <th>Image</th>
                        </tr>
                    </thead>
                    <tbody id="tableBody">
                        <tr><td class="empty" colspan="5">No rows loaded yet.</td></tr>
                    </tbody>
                </table>
            </div>
        </section>
    </div>

    <script>
        let currentPage = 1;
        let totalMatches = 0;

        const loadButton = document.getElementById("loadButton");
        const prevButton = document.getElementById("prevButton");
        const nextButton = document.getElementById("nextButton");
        const tableBody = document.getElementById("tableBody");
        const summary = document.getElementById("summary");
        const pageIndicator = document.getElementById("pageIndicator");
        const loadingState = document.getElementById("loadingState");
        const universityInput = document.getElementById("universityInput");
        const searchInput = document.getElementById("searchInput");
        const pageSizeInput = document.getElementById("pageSizeInput");
        const statusFilter = document.getElementById("statusFilter");

        function escapeHtml(value) {
            const div = document.createElement("div");
            div.textContent = value ?? "";
            return div.innerHTML;
        }

        function setLoading(isLoading) {
            loadingState.style.display = isLoading ? "block" : "none";
            loadButton.disabled = isLoading;
            prevButton.disabled = isLoading || currentPage <= 1;
            nextButton.disabled = isLoading;
        }

        function renderRows(rows) {
            if (!rows.length) {
                tableBody.innerHTML = '<tr><td class="empty" colspan="5">No matching image records found.</td></tr>';
                return;
            }

            tableBody.innerHTML = rows.map((row) => {
                const imageUrl = escapeHtml(row.image_url);
                const status = row.processed_status === "done" ? "done" : "pending";
                return `
                    <tr>
                        <td class="name-cell">
                            <strong>${escapeHtml(row.name)}</strong>
                            <span class="subtle">${escapeHtml(row.profile_id)}</span>
                        </td>
                        <td><span class="pill ${status}">${escapeHtml(status)}</span></td>
                        <td>${escapeHtml(row.university)}</td>
                        <td><a class="image-link" href="${imageUrl}" target="_blank" rel="noreferrer">${imageUrl}</a></td>
                        <td><img class="thumb" src="${imageUrl}" alt="${escapeHtml(row.name)}"></td>
                    </tr>
                `;
            }).join("");
        }

        async function loadRows(page = 1) {
            currentPage = page;
            setLoading(true);

            const university = universityInput.value.trim();
            const search = searchInput.value.trim();
            const processedStatus = statusFilter.value;
            const pageSize = Math.max(1, Math.min(200, parseInt(pageSizeInput.value || "50", 10)));
            const params = new URLSearchParams({
                university,
                search,
                processed_status: processedStatus,
                page: String(currentPage),
                page_size: String(pageSize),
            });

            try {
                const response = await fetch(`/api/faculty-images?${params.toString()}`);
                if (!response.ok) {
                    const payload = await response.json();
                    throw new Error(payload.detail || "Failed to fetch faculty images");
                }

                const data = await response.json();
                totalMatches = data.total_matches || 0;
                renderRows(data.rows || []);
                const doneCount = (data.rows || []).filter((row) => row.processed_status === "done").length;
                const pendingCount = (data.rows || []).filter((row) => row.processed_status !== "done").length;

                const pageStart = totalMatches === 0 ? 0 : ((data.page - 1) * data.page_size) + 1;
                const pageEnd = totalMatches === 0 ? 0 : Math.min(data.page * data.page_size, totalMatches);
                const statusLabel = processedStatus === "all"
                    ? "all statuses"
                    : processedStatus === "done"
                        ? "done only"
                        : "unprocessed only";
                summary.textContent = `Showing ${pageStart}-${pageEnd} of ${totalMatches} matching image records for ${statusLabel}. This page: ${doneCount} done, ${pendingCount} pending.`;
                pageIndicator.textContent = `Page ${data.page}`;
                prevButton.disabled = data.page <= 1;
                nextButton.disabled = pageEnd >= totalMatches;
            } catch (error) {
                tableBody.innerHTML = `<tr><td class="empty" colspan="5">${escapeHtml(error.message || "Unknown error")}</td></tr>`;
                summary.textContent = "Request failed.";
                pageIndicator.textContent = `Page ${currentPage}`;
                prevButton.disabled = currentPage <= 1;
                nextButton.disabled = true;
            } finally {
                setLoading(false);
            }
        }

        loadButton.addEventListener("click", () => loadRows(1));
        prevButton.addEventListener("click", () => {
            if (currentPage > 1) {
                loadRows(currentPage - 1);
            }
        });
        nextButton.addEventListener("click", () => {
            const pageSize = Math.max(1, Math.min(200, parseInt(pageSizeInput.value || "50", 10)));
            if (currentPage * pageSize < totalMatches) {
                loadRows(currentPage + 1);
            }
        });

        searchInput.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                loadRows(1);
            }
        });

        universityInput.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                loadRows(1);
            }
        });

        statusFilter.addEventListener("change", () => loadRows(1));

        loadRows(1);
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)
