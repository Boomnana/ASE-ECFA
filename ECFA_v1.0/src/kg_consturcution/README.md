# kg_consturcution（缺陷知识图谱构建流水线）

本目录提供一条“从缺陷报告文本 → 三元组抽取 → 同义实体对齐 → 生成图节点/边 →（可选）导入 Neo4j”的端到端脚本流水线，主要面向中文 App 缺陷报告场景，实体/关系体系与下游 `defect_subgraph_cover`、`defect_rag` 能直接对接。

## 目录结构

```
kg_consturcution/
├── Step1_triple_extractor.py          # LLM 抽取实体与三元组（从缺陷报告文本）
├── Step2_synonym_alignment.py         # LLM 实体同义对齐（分桶→合并→验证→聚类）
├── Step3_build_kg.py                  # 用对齐结果替换三元组并生成 nodes.xlsx / edges.xlsx
├── Step4_neo4j_importer.py            # 导入 Neo4j（读 nodes.xlsx / edges.xlsx）
├── llm_client.py                      # LLM 配置与调用封装（ZhipuAI via LangChain）
├── prompt/
│   ├── triple_extraction_prompt.py    # 抽取提示词（实体定义+7种关系约束）
│   └── entity_alignment_prompt.py     # 对齐提示词（概念桶/候选对/验证）
├── entity_alignment/
│   ├── stage1_bucketing.py            # 阶段1：语义分桶
│   ├── stage2_merging.py              # 阶段2：桶间合并（可选多轮）
│   ├── stage3_alignment.py            # 阶段3：桶内候选对生成+批量验证
│   ├── stage4_clustering.py           # 阶段4：并查集闭包与最终聚类
│   └── common.py                      # 输入读取/标准化/粗分块/few-shot 加载
└── utils/
    ├── llm_conversation_logger.py     # 记录 LLM 交互（用于复盘与成本追踪）
    ├── sentence_utils.py              # 清洗 LLM 输出（提升 JSON 解析成功率）
```

> 注意：仓库 `.gitignore` 默认忽略 `src/kg_consturcution/output` 与 `src/kg_consturcution/graph_output`，这些目录通常是运行产物。

## 实体与关系体系（约束非常重要）

实体类型（中文）在 `prompt/triple_extraction_prompt.py` 中定义，核心包括：
- 功能模块（MOD）
- 用户操作（OP）
- 影响元素（ELEM）
- 用户感知现象（PHEN）
- 系统诊断信息（DIAG）
- 问题陈述（ISSUE）

关系类型严格限制为 7 种（在 `prompt/triple_extraction_prompt.py` 内声明），用于保持结构可控、便于下游做覆盖与证据引用：
- `located_in`（ELEM → MOD）
- `performs_on`（OP → ELEM）
- `triggers`（OP → MOD）
- `results_in`（OP → PHEN）
- `accompanied_by`（PHEN ↔ DIAG）
- `concerns`（ISSUE → MOD）
- `has_symptom`（ISSUE → PHEN）

如果模型输出不在此集合内，建议在抽取侧修正（而不是在构图侧“硬接”新关系），否则下游链路（覆盖/事件链）会变得不可控。

## 运行前准备

### 1) Python 环境与依赖

本目录脚本依赖 `pandas`、`openpyxl`、`langchain`、`langchain_community`、`httpx`、`tenacity`、`tqdm` 等；Neo4j 导入还需要 `neo4j`、`python-dotenv`、`jieba`。

项目未在仓库内固定 requirements/pyproject，因此建议在你现有的 conda/venv 环境里按报错缺什么装什么，或基于你们内部的依赖管理方式补齐。

### 2) LLM 配置（必需）

`Step1_triple_extractor.py` / `Step2_synonym_alignment.py` 默认读取仓库根目录的 `LLM-config.yaml`（见 `llm_client.py:21-47`），其结构为：

```yaml
model:
  key: "YOUR_API_KEY"
  name: "YOUR_MODEL_NAME"
  temperature: 0.2
  max_tokens: 5000
  concurrency: 5
  timeout: 30
  max_retries: 2
```

`llm_client.py` 会把 `key` 注入到环境变量 `ZHIPUAI_API_KEY`（见 `llm_client.py:107-112`）。

### 3) Neo4j 环境（可选）

仅在你要跑 `Step4_neo4j_importer.py` 时需要：
- 本机已启动 Neo4j，且可通过 `bolt://localhost:7687` 访问（默认值）
- 在仓库根目录 `.env` 或系统环境变量中设置：
  - `NEO4J_URI`
  - `NEO4J_USER`
  - `NEO4J_PASSWORD`（必需；未设置会直接抛错，见 `Step4_neo4j_importer.py:46-48`）

## 快速开始（推荐工作流）

下面的流程强调“路径对齐”，因为 Step3 默认只读取固定路径 `kg_consturcution/output/triples.xlsx` 与 `kg_consturcution/output/aligned_clusters.xlsx`（见 `Step3_build_kg.py:8-10`）。

### Step 1：从缺陷报告抽取三元组

输入：一个 Excel（包含 `description` 文本列，建议有 `id` 作为 report_id）。

示例命令（把三元组输出到固定位置，方便 Step2/Step3 接续）：

```powershell
python src/kg_consturcution/Step1_triple_extractor.py `
  --input-file input/6已标注APP.xlsx `
  --sheet-name 雅思听力 `
  --triple-path src/kg_consturcution/output/triples.xlsx `
  --output-file src/kg_consturcution/output/kg_extraction_results.xlsx
```

输出：
- `output/kg_extraction_results.xlsx`：逐 report 的抽取结果与状态（成功/失败/无三元组）
- `output/triples.xlsx`：平铺三元组表（供 Step2/Step3 使用）

`triples.xlsx` 关键列（见 `Step1_triple_extractor.py:222-243`）：
- `subject`, `subject_type`
- `relation`
- `object`, `object_type`
- `source_row_index`：report_id（证据锚点）
- `raw_input`：原始文本（可用于审计/回放）

### Step 2：实体同义对齐（可选但强烈建议）

作用：减少同义表述导致的“节点碎片化”，例如“登录页/登录界面/登录页面”。

示例命令：

```powershell
python src/kg_consturcution/Step2_synonym_alignment.py `
  --input-file src/kg_consturcution/output/triples.xlsx `
  --stage2-iterative `
  --stage2-max-rounds 3
```

输出（见 `Step2_synonym_alignment.py:37-40, 223-227, 260-267`）：
- `output/aligned_clusters.xlsx`：最终聚类（Step3 会读取这个固定路径）
- `output/alignment_files_YYYYmmdd_HHMMSS/`：本次运行的详细中间产物
  - `semantic_buckets.json`（阶段1）
  - `stage2_merged_buckets.json`（阶段2）
  - `alignment_pairs.json`（阶段3）
  - `stage4_final_clusters.json`（阶段4）
  - `stage1_batches.json`（调试辅助）

重要参数：
- `--entity-type-filter`：仅对某些类型的实体做对齐（避免跨类型混桶），类型值必须与 `triples.xlsx` 的 `subject_type/object_type` 一致（见 `entity_alignment/common.py:20-49`）
- `--coarse-size`：阶段1 粗分块大小，影响提示长度与调用次数（见 `Step2_synonym_alignment.py:44-45`）
- `--bucket-limit`：桶内实体上限，过大会导致对齐提示过长（见 `Step2_synonym_alignment.py:49-50`）

### Step 3：构建图节点与边（nodes.xlsx / edges.xlsx）

作用：
1) 用对齐结果把 `triples.xlsx` 中的实体替换成“聚类代表词”（生成 `optimized_triplets.xlsx`）
2) 生成带 ID 的图节点与边，写入 `graph_output/`

示例命令：

```powershell
python src/kg_consturcution/Step3_build_kg.py
```

输出（见 `Step3_build_kg.py:10-12, 133-142`）：
- `output/optimized_triplets.xlsx`
- `graph_output/nodes.xlsx`（列：`id,name,type,source_row_index`）
- `graph_output/edges.xlsx`（列：`id,source_id,target_id,relation,source_type,target_type,source_row_index`）

可调参数（见 `Step3_build_kg.py:13-16`）：
- `--allowed-types`：保留哪些类型进入图（默认六大类）
- `--node-id-prefix` / `--node-id-width`：节点 ID 格式（默认 `1011 + 6位`）

### Step 4：导入 Neo4j（可选）

作用：把 `graph_output/nodes.xlsx` 与 `graph_output/edges.xlsx` 导入 Neo4j，并创建索引/约束。

在设置好 `.env` 后运行：

```powershell
python src/kg_consturcution/Step4_neo4j_importer.py
```

该脚本会强校验 `NEO4J_PASSWORD`，并为每个实体类型建立唯一约束（见 `Step4_neo4j_importer.py:150-162`）。

## 常见问题（Troubleshooting）

- Step1 解析失败 / JSON 不合法：通常是模型输出格式漂移，优先检查 `output/*/Step1_log_extractor.txt` 与 `utils/sentence_utils.clean_llm_output` 的清洗效果。
- Step2 对齐成本高/速度慢：先减小 `--coarse-size`、`--bucket-limit`，并限定 `--entity-type-filter`，避免跨类型实体一起对齐。
- Step3 找不到输入文件：确认 `src/kg_consturcution/output/triples.xlsx` 与 `src/kg_consturcution/output/aligned_clusters.xlsx` 都存在（Step3 读取路径是固定的）。
- Step4 连接 Neo4j 失败：确认 Neo4j 已启动，`NEO4J_URI` 可访问，且 `.env` 中已设置 `NEO4J_PASSWORD`。

