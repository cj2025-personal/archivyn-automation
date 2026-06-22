"""
FastAPI application main file
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import (
    admin_scripts,
    batch,
    chunks,
    clean,
    enrichment_viewer,
    extract,
    faculty_images,
    merge,
    process,
    profiles,
    view_db,
)
from api.services.admin_script_platform import start_admin_script_platform, stop_admin_script_platform
import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

# Load .env file from project root (parent of api folder)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Note: nest_asyncio is NOT applied here because it can interfere with Playwright on Windows
# Playwright requires SelectorEventLoop, and nest_asyncio wrapping ProactorEventLoop causes issues
# The event loop policy is set in start_server.py before uvicorn starts


@asynccontextmanager
async def lifespan(_app: FastAPI):
    start_admin_script_platform()
    yield
    stop_admin_script_platform()


app = FastAPI(
    title="Faculty Data Extraction API",
    description="API for extracting and processing faculty profile data",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(extract.router)
app.include_router(process.router)
app.include_router(merge.router)
app.include_router(admin_scripts.router)
app.include_router(admin_scripts.portal_router)
app.include_router(batch.router)
app.include_router(view_db.router)
app.include_router(clean.router)
app.include_router(chunks.router)
app.include_router(profiles.router)
app.include_router(faculty_images.router)
app.include_router(enrichment_viewer.router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Faculty Data Extraction API",
        "version": "1.0.0",
        "endpoints": {
            "extract": "/api/extract-all",
            "process": "/api/process-resource",
            "merge": "/api/merge-and-store",
            "automation_module": "/admin/automation",
            "admin_scripts_catalog": "/api/admin/scripts/catalog",
            "admin_script_modules": "/api/admin/scripts/modules",
            "batch": "/api/batch-process",
            "batch_interface": "/api/batch-interface",
            "view_db": "/api/view-db",
            "view_db_interface": "/api/view-db-interface",
            "clean": "/clean/batch",
            "pending_count": "/clean/pending-count",
            "enrichment_dashboard": "/profiles/",
            "enrichment_detail": "/profiles/{profile_id}",
            "enrichment_professors_api": "/profiles/api/professors",
            "enrichment_professor_api": "/profiles/api/professor/{profile_id}",
            "faculty_images_api": "/api/faculty-images",
            "faculty_images_view": "/faculty-images",
            "enrichment_viewer": "/enrichment",
            "enrichment_api_professors": "/api/enrichment/professors",
            "enrichment_api_detail": "/api/enrichment/professor/{profile_id}",
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}
