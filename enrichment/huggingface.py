"""
Hugging Face Hub collector — models / datasets published by the researcher.

API docs: https://huggingface.co/docs/hub/api
Free; public endpoints don't require auth. HF_TOKEN raises rate limits.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .base_collector import BaseCollector, CollectorResult, ProfessorQuery

BASE = "https://huggingface.co/api"


class HuggingFaceCollector(BaseCollector):
    def __init__(self, **kwargs):
        kwargs.setdefault("rate_limit_delay", 0.8)
        super().__init__(**kwargs)
        self.token = os.getenv("HF_TOKEN", "")

    @property
    def source_name(self) -> str:
        return "huggingface"

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    async def _hf_get(self, url: str, params: Optional[Dict] = None) -> Optional[Any]:
        await self._rate_limit()
        try:
            client = await self.get_client()
            resp = await client.get(url, params=params, headers=self._headers())
            if resp.status_code in (401, 403, 404):
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    async def collect(self, query: ProfessorQuery) -> CollectorResult:
        # HF search is by full-text; pass name.
        models = await self._hf_get(f"{BASE}/models", params={"search": query.name, "limit": 30}) or []
        datasets = await self._hf_get(f"{BASE}/datasets", params={"search": query.name, "limit": 30}) or []

        q_last = query.last_name.lower()
        q_first = query.first_name.lower()

        def matches(item: Dict) -> bool:
            author = (item.get("author") or "").lower()
            model_id = (item.get("id") or item.get("modelId") or "").lower()
            if q_last and (q_last in author or q_last in model_id):
                return True
            if q_first and q_first in author:
                return True
            return False

        filt_models = [m for m in models if matches(m)]
        filt_datasets = [d for d in datasets if matches(d)]

        if not filt_models and not filt_datasets:
            return self._make_result(query, success=False, error="No HF models/datasets matched")

        data = {
            "total_models": len(filt_models),
            "total_datasets": len(filt_datasets),
            "models": [
                {
                    "id": m.get("id"),
                    "author": m.get("author"),
                    "downloads": m.get("downloads"),
                    "likes": m.get("likes"),
                    "tags": m.get("tags"),
                    "pipeline_tag": m.get("pipeline_tag"),
                    "url": f"https://huggingface.co/{m.get('id')}" if m.get("id") else None,
                }
                for m in filt_models[:25]
            ],
            "datasets": [
                {
                    "id": d.get("id"),
                    "author": d.get("author"),
                    "downloads": d.get("downloads"),
                    "likes": d.get("likes"),
                    "tags": d.get("tags"),
                    "url": f"https://huggingface.co/datasets/{d.get('id')}" if d.get("id") else None,
                }
                for d in filt_datasets[:25]
            ],
        }
        return self._make_result(query, success=True, data=data, raw_text=self._to_text(data, query))

    def _to_text(self, data: Dict, query: ProfessorQuery) -> str:
        lines = [f"=== Hugging Face artifacts for {query.name} ==="]
        lines.append(f"Models: {data['total_models']} | Datasets: {data['total_datasets']}")
        lines.append("")
        if data["models"]:
            lines.append("── Models ──")
            for m in data["models"]:
                lines.append(
                    f"• {m['id']} — downloads={m.get('downloads')} likes={m.get('likes')} "
                    f"task={m.get('pipeline_tag')}"
                )
        if data["datasets"]:
            lines.append("\n── Datasets ──")
            for d in data["datasets"]:
                lines.append(f"• {d['id']} — downloads={d.get('downloads')} likes={d.get('likes')}")
        return "\n".join(lines)
