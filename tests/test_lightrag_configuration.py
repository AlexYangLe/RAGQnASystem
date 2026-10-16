from pathlib import Path


def test_lightrag_uses_only_supported_default_storage_backends():
    source = Path("webui.py").read_text(encoding="utf-8")

    assert 'kv_storage="Neo4jStorage"' not in source
    assert 'graph_storage="Neo4jStorage"' not in source


def test_webui_uses_the_initialized_async_lightrag_runtime():
    source = Path("webui.py").read_text(encoding="utf-8")

    assert "AsyncLightRAGClient" in source
    assert "lightrag_config" in source
    assert "get_lightrag_client().query_local_context(query)" in source
    assert "rag_client.query(" not in source
    assert "rag_client = get_lightrag_client()" not in source


def test_webui_does_not_keep_the_legacy_ner_or_intent_pipeline():
    source = Path("webui.py").read_text(encoding="utf-8")

    assert "Intent_Recognition" not in source
    assert "load_model(" not in source
    assert "intent_router" not in source


def test_public_repository_does_not_embed_a_default_admin_password():
    source = Path("user_data_storage.py").read_text(encoding="utf-8")

    assert 'Credentials("admin", "admin123"' not in source
    assert "INITIAL_ADMIN_USERNAME" in source
    assert "INITIAL_ADMIN_PASSWORD" in source
