import asyncio

from lightrag_runtime import AsyncLightRAGClient, compact_context, find_exact_disease_context


class FakeRAG:
    def __init__(self):
        self.initialized = False
        self.finalized = False
        self.query = None

    async def initialize_storages(self):
        self.initialized = True

    async def aquery(self, query, param):
        self.query = (query, param.mode, param.only_need_context)
        return "苯中毒的图谱上下文"

    async def finalize_storages(self):
        self.finalized = True


def test_client_initializes_and_queries_on_one_async_runtime():
    rag = FakeRAG()
    client = AsyncLightRAGClient(lambda: rag)
    try:
        assert client.query_local_context("苯中毒的症状") == "苯中毒的图谱上下文"
        assert rag.initialized
        assert rag.query == ("苯中毒的症状", "local", True)
    finally:
        client.close()

    assert rag.finalized


class FakeRAGWithWorker(FakeRAG):
    async def initialize_storages(self):
        await super().initialize_storages()
        self.worker = asyncio.create_task(asyncio.sleep(60))


def test_client_cancels_background_tasks_before_closing_its_loop():
    rag = FakeRAGWithWorker()
    client = AsyncLightRAGClient(lambda: rag)

    client.close()

    assert rag.worker.cancelled()


def test_exact_disease_match_returns_the_imported_disease_chunk():
    benzene_chunk = "疾病名称：苯中毒\n症状：恶心；抽搐；感觉障碍"
    other_chunk = "疾病名称：肺炎\n症状：咳嗽"

    context = find_exact_disease_context("苯中毒的症状有哪些？", [other_chunk, benzene_chunk])

    assert context == benzene_chunk


def test_fallback_context_is_capped_to_fit_the_answer_model_window():
    assert compact_context("x" * 5000, max_characters=3000) == "x" * 3000
