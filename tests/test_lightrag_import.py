from lightrag_import import (
    custom_kg_from_records,
    lightrag_config,
    load_first_records,
    record_to_document,
    source_paths,
)


def test_first_fifty_records_are_json_lines_and_receive_unique_source_paths():
    records = load_first_records("data/medical_new_2.json", limit=50)

    assert len(records) == 50
    assert records[0]["name"] == "肺泡蛋白质沉积症"
    assert records[-1]["name"]
    assert len(set(source_paths("data/medical_new_2.json", len(records)))) == 50


def test_record_conversion_keeps_named_medical_fields():
    document = record_to_document(
        {
            "name": "示例疾病",
            "desc": "示例简介",
            "symptom": ["发热", "咳嗽"],
            "check": ["血常规"],
        }
    )

    assert "疾病名称：示例疾病" in document
    assert "疾病简介：示例简介" in document
    assert "症状：发热；咳嗽" in document
    assert "检查项目：血常规" in document


def test_import_config_matches_the_single_ollama_inference_slot():
    assert lightrag_config()["llm_model_max_async"] == 1


def test_structured_records_become_lightrag_chunks_entities_and_relations():
    kg = custom_kg_from_records(
        [{"name": "示例疾病", "desc": "示例简介", "symptom": ["发热"], "check": ["血常规"]}],
        "data/medical_new_2.json",
    )

    assert len(kg["chunks"]) == 1
    assert {entity["entity_name"] for entity in kg["entities"]} >= {"示例疾病", "发热", "血常规"}
    assert {relation["keywords"] for relation in kg["relationships"]} >= {"疾病的症状", "疾病所需检查"}
