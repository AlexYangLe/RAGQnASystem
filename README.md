# 🏥 RAGQnASystem

> 基于 **LightRAG + Ollama + Streamlit** 的本地医疗知识问答原型。

本项目使用 LightRAG 将医疗资料转换为可检索的文本分块、实体和关系，通过本地 Ollama 模型生成基于知识库上下文的回答。

> **医疗安全声明**：本项目仅用于技术演示、资料检索和学习，不提供诊断、处方、剂量建议或急救决策。任何用于真实场景的知识库均须由专业人员审核、标注来源并定期更新。

## 特性

- 本地运行：Ollama、LightRAG 与 Streamlit 均可在本机运行。
- 无需运行时 Neo4j：默认使用 LightRAG 本地文件存储。
- 无需 BERT/NER 权重：知识检索不依赖旧版命名实体识别模型。
- 结构化导入：将 JSON Lines 医疗记录直接转换为知识分块、实体和关系。
- 可追溯调试：管理员可查看 LightRAG 召回的上下文。
- 流式回答：通过 Ollama 在页面中逐步输出回答。

## 架构

```text
data/medical_new_2.json（JSON Lines 医疗数据）
                  │
                  ▼
          lightrag_import.py
      ┌───────────┼────────────────────┐
      ▼           ▼                    ▼
   文档分块      实体                  关系
      └───────────┴────────────────────┘
                  ▼
     medical_lightrag_db/（本地知识库）
                  │
用户问题 → Streamlit → AsyncLightRAGClient → LightRAG 召回上下文
                                               │
                                               ▼
                                     Ollama / qwen2.5:7b
                                               │
                                               ▼
                                           流式答案
```

## 技术栈

| 组件 | 用途 |
|---|---|
| Python 3.10+ | 运行环境 |
| Streamlit | 登录与问答页面 |
| LightRAG | 本地知识图谱与向量检索 |
| Ollama | 本地大语言模型和嵌入模型服务 |
| `qwen2.5:7b` | 回答生成与 LightRAG LLM 调用 |
| `nomic-embed-text:latest` | 向量嵌入，维度为 768 |

## 快速开始

### 1. 克隆并创建环境

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RAGQnASystem

conda create -n ragqna python=3.10 -y
conda activate ragqna

pip install -r requirements.txt
```

### 2. 安装并启动 Ollama

从 [Ollama 官网](https://ollama.com/download) 安装应用。启动 Ollama App，或在独立终端执行：

```bash
ollama serve
```

再拉取本项目需要的模型：

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text:latest
ollama list
```

确认本地 API 已启动：

```bash
curl http://127.0.0.1:11434/api/tags
```

若此命令无法连接，页面会出现 `ConnectionError: Failed to connect to Ollama`。模型已下载并不等于 Ollama API 服务已启动。

### 3. 导入测试知识库

项目内置的数据源 `data/medical_new_2.json` 为 JSON Lines 格式。首次建议只导入前 50 条记录：

```bash
python lightrag_import.py \
  --data-path data/medical_new_2.json \
  --working-dir medical_lightrag_db \
  --limit 50
```

导入脚本会把疾病名称、简介、病因、症状、预防、检查、药品、科室和治疗方式等字段转换为 LightRAG 的分块、实体与关系。

成功后应生成：

```text
medical_lightrag_db/
├── graph_chunk_entity_relation.graphml
├── kv_store_text_chunks.json
├── vdb_chunks.json
├── vdb_entities.json
└── vdb_relationships.json
```

### 4. 启动问答页面

```bash
streamlit run login.py
```

打开终端显示的地址，通常是 <http://localhost:8501>。

导入前 50 条后，可使用下列问题验证：

```text
苯中毒的症状有哪些？
百日咳的预防措施是什么？
```

管理员账户可展开“知识库信息”确认召回上下文；正常情况下该区域不应显示 `None`。

## 项目结构

```text
RAGQnASystem/
├── login.py                 # Streamlit 登录入口
├── webui.py                 # 页面编排、召回上下文和流式回答
├── lightrag_import.py       # JSON Lines 医疗数据导入器
├── lightrag_runtime.py      # LightRAG 异步初始化与同步查询适配器
├── config.py                # 环境配置
├── data/
│   └── medical_new_2.json   # 内置医疗数据源
├── medical_lightrag_db/     # 本地索引（运行生成，不提交）
└── tests/                   # 导入与运行时测试
```

## 关键配置

| 配置 | 默认值 | 说明 |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API 地址 |
| `OLLAMA_QWEN_MODEL` | `qwen2.5:7b` | 问答使用的本地模型 |
| 向量模型 | `nomic-embed-text:latest` | 导入与查询必须一致 |
| 向量维度 | `768` | 对应 `nomic-embed-text` |
| 知识库目录 | `medical_lightrag_db` | 导入与问答页必须一致 |
| `INITIAL_ADMIN_USERNAME` | 未设置 | 可选：首次启动时创建管理员用户名 |
| `INITIAL_ADMIN_PASSWORD` | 未设置 | 可选：首次启动时创建管理员密码 |

若更换向量模型或维度，必须删除并重新建立 `medical_lightrag_db/`，不能与旧索引混用。

首次启动时可直接在页面注册普通用户。若需使用管理员的检索调试面板，请在启动前设置两个管理员环境变量：

```bash
export INITIAL_ADMIN_USERNAME='admin'
export INITIAL_ADMIN_PASSWORD='请改成一个强密码'
streamlit run login.py
```

本地凭证文件会写入 `tmp_data/user_credentials.json`，该文件已被 Git 忽略，不会提交到 GitHub。当前登录模块仍是本地演示实现，密码以明文保存在本机；不要用于公开部署或真实用户账号。

## 扩展知识库

小样本验证通过后，可以逐步扩展导入量：

```bash
python lightrag_import.py --limit 500
```

首次不建议直接导入全部数据。请先检查数据质量、召回结果与异常处理，再逐步扩大规模。导入临床指南、药品说明书或论文时，建议保留来源链接、发布日期与审核日期，建立可追溯的 Markdown 数据源后再接入导入器。

## 不再需要的旧组件

当前版本已经移除 Neo4j、BERT NER、NER 模型权重、固定 Cypher 查询、固定意图路由和大模型微调示例。它们都不是部署前置条件，也不应与 `medical_lightrag_db/` 混用。

若未来确实需要独立维护大规模图谱或执行 Cypher 分析，应在单独分支中重新设计图存储适配层并补齐回归测试，而不是直接恢复旧代码。

## GitHub 发布注意事项

不要提交本地模型、索引缓存、账号密码或环境变量。建议在 `.gitignore` 中加入：

```gitignore
medical_lightrag_db/
tmp_data/user_credentials.json
.env
.DS_Store
node_modules/
__pycache__/
.pytest_cache/
```

发布前请在干净环境中验证完整流程：安装依赖 → 启动 Ollama → 拉取两个模型 → 导入 50 条 → 启动 Streamlit → 查询“苯中毒的症状”。

## 已知限制

- 内置医疗资料未经临床审核，不能替代专业医疗建议。
- 仅导入前 50 条时，其他疾病不在知识库覆盖范围内。
- 本地推理速度和效果取决于机器内存、CPU/GPU 和 Ollama 模型配置。
- 目前采用本地文件型 LightRAG 存储；不提供跨机器共享索引或多人并发写入能力。

## 许可证

请在仓库根目录保留并明确适用的许可证，例如 MIT。
