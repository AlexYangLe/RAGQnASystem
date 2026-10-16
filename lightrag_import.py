"""Convert the medical JSONL dataset into documents and import them into LightRAG."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any


FIELD_LABELS = {
    "name": "疾病名称",
    "desc": "疾病简介",
    "category": "分类",
    "prevent": "预防措施",
    "cause": "病因",
    "symptom": "症状",
    "yibao_status": "医保状态",
    "get_prob": "发病概率",
    "easy_get": "易感人群",
    "get_way": "传播途径",
    "acompany": "并发疾病",
    "cure_department": "就诊科室",
    "cure_way": "治疗方法",
    "cure_lasttime": "治疗周期",
    "cured_prob": "治愈概率",
    "common_drug": "常用药品",
    "cost_money": "治疗费用",
    "check": "检查项目",
    "do_eat": "宜吃食物",
    "not_eat": "忌吃食物",
    "recommand_eat": "推荐食谱",
    "recommand_drug": "推荐药品",
    "drug_detail": "药品与生产商",
}


def load_first_records(data_path: str | Path, limit: int) -> list[dict[str, Any]]:
    """Read the first ``limit`` records from the dataset's JSON Lines format."""
    records: list[dict[str, Any]] = []
    with Path(data_path).open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip().rstrip(",")
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是有效 JSON：{exc}") from exc
            if len(records) == limit:
                return records
    raise ValueError(f"仅解析到 {len(records)} 条记录，少于请求的 {limit} 条。")


def record_to_document(record: dict[str, Any]) -> str:
    """Render one disease record as a labeled Chinese document for retrieval."""
    if not record.get("name"):
        raise ValueError("记录缺少疾病名称。")

    parts: list[str] = []
    for field, label in FIELD_LABELS.items():
        value = record.get(field)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            value = "；".join(str(item) for item in value if item not in (None, ""))
        elif isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        parts.append(f"{label}：{value}")
    return "\n".join(parts)


def source_paths(data_path: str | Path, count: int) -> list[str]:
    """Give each JSONL record a unique source identity for LightRAG de-duplication."""
    source = str(data_path)
    return [f"{source}.record-{index:04d}" for index in range(1, count + 1)]


def lightrag_config() -> dict[str, Any]:
    """Return storage settings compatible with this machine's single Ollama slot."""
    return {
        "llm_model_name": "qwen2.5:7b",
        "llm_model_max_async": 1,
        "max_parallel_insert": 1,
        "embedding_func_max_async": 1,
        "chunk_token_size": 600,
        "chunk_overlap_token_size": 50,
    }


RELATION_FIELDS = (
    ("symptom", "疾病的症状", "症状"),
    ("common_drug", "疾病使用药品", "药品"),
    ("recommand_drug", "疾病使用药品", "药品"),
    ("do_eat", "疾病宜吃食物", "食物"),
    ("recommand_eat", "疾病宜吃食物", "食物"),
    ("not_eat", "疾病忌吃食物", "食物"),
    ("check", "疾病所需检查", "检查项目"),
    ("cure_department", "疾病所属科目", "科目"),
    ("cure_way", "治疗的方法", "治疗方法"),
    ("acompany", "疾病并发疾病", "疾病"),
)


def _values(value: Any) -> list[str]:
    if not isinstance(value, list):
        value = [value]
    result = []
    for item in value:
        if isinstance(item, list):
            result.extend(_values(item))
        elif item not in (None, ""):
            result.append(str(item).removesuffix("..."))
    return result


def custom_kg_from_records(records: list[dict[str, Any]], data_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Convert structured medical records directly to LightRAG's custom KG shape."""
    chunks: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        disease = str(record["name"])
        source_id = f"medical-new-2-source-{index:04d}"
        file_path = f"{data_path}.record-{index:04d}"
        document = record_to_document(record)
        chunks.append({"content": document, "source_id": source_id, "file_path": file_path, "chunk_order_index": 0})
        entities.append({
            "entity_name": disease,
            "entity_type": "疾病",
            "description": document,
            "source_id": source_id,
            "file_path": file_path,
        })
        for field, relation, entity_type in RELATION_FIELDS:
            for target in _values(record.get(field, [])):
                if not target:
                    continue
                entities.append({
                    "entity_name": target,
                    "entity_type": entity_type,
                    "description": f"{entity_type}：{target}",
                    "source_id": source_id,
                    "file_path": file_path,
                })
                relationships.append({
                    "src_id": disease,
                    "tgt_id": target,
                    "description": f"{disease}{relation}{target}。",
                    "keywords": relation,
                    "source_id": source_id,
                    "file_path": file_path,
                    "weight": 1.0,
                })
    return {"chunks": chunks, "entities": entities, "relationships": relationships}


async def import_records(data_path: str, working_dir: str, limit: int) -> dict[str, Any]:
    """Convert and import the requested leading JSONL records into LightRAG."""
    import numpy as np
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_model_complete
    from lightrag.utils import EmbeddingFunc
    from ollama import AsyncClient

    records = load_first_records(data_path, limit)
    custom_kg = custom_kg_from_records(records, data_path)

    async def ollama_embed(texts: list[str], **_: Any) -> np.ndarray:
        client = AsyncClient(host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
        response = await client.embed(model="nomic-embed-text:latest", input=texts)
        embeddings = response.get("embeddings", [])
        if len(embeddings) != len(texts):
            raise RuntimeError(f"向量数量不匹配：请求 {len(texts)}，返回 {len(embeddings)}。")
        return np.asarray(embeddings)

    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=EmbeddingFunc(embedding_dim=768, max_token_size=8192, func=ollama_embed),
        llm_model_func=ollama_model_complete,
        **lightrag_config(),
    )

    await rag.initialize_storages()
    try:
        await rag.ainsert_custom_kg(custom_kg)
        return {
            "documents_submitted": len(custom_kg["chunks"]),
            "entities_submitted": len(custom_kg["entities"]),
            "relationships_submitted": len(custom_kg["relationships"]),
            "first_disease": records[0]["name"],
            "last_disease": records[-1]["name"],
        }
    finally:
        await rag.finalize_storages()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导入 JSONL 医疗知识到 LightRAG")
    parser.add_argument("--data-path", default="data/medical_new_2.json")
    parser.add_argument("--working-dir", default="medical_lightrag_db")
    parser.add_argument("--limit", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = asyncio.run(import_records(args.data_path, args.working_dir, args.limit))
    print(json.dumps(result, ensure_ascii=False))
