# 面向中文缺陷报告的知识图谱构建与子图覆盖：技术文档（论文体）

## 摘要
本文描述一个端到端的缺陷分析流水线：输入中文 App 缺陷报告文本，利用受控的 LLM 信息抽取生成实体与关系三元组，结合基于 LLM 的同义实体对齐降低图谱碎片化，进而构建带证据锚点（report_id）的缺陷知识图谱。基于该图谱，系统提出一种“报告诱导子图 + 语义合并 + 贪心集合覆盖”的方法，自动生成少量代表性子图以覆盖主要故障模式，并输出可读的 Excel 报告与 JSON/摘要 JSON。文档给出算法原理、执行流程、关键实现、实验设置建议、参数敏感性、复杂度分析与局限性。

## 1. 引言
缺陷报告通常以自然语言记录故障触发条件、现象与诊断信息，存在描述随意、同义表达多、结构化程度低的问题，使得缺陷聚类、归因分析与覆盖统计困难。本项目将缺陷报告转化为结构化知识图谱，并进一步在图上抽取可解释的代表子图集合，用于“覆盖主要缺陷模式”的分析与汇总。

本项目在 `src/` 下由两个核心子模块构成：
- 知识图谱构建流水线：`src/kg_consturcution/`（Step1–Step4）
- 子图覆盖与报表输出：`src/defect_subgraph_cover/`

## 2. 方法概述

### 2.1 任务定义
给定缺陷报告集合 \(\mathcal{D}=\{d_i\}\)，每条报告包含唯一标识 `id`（或行号）与文本 `description`。目标为：
1. 构建一个带类型的实体集合与关系集合组成的图 \(G=(V,E)\)，且每个节点/边都保留“证据锚点”——其来源报告集合 `report_ids`；
2. 从图中生成一组候选子图 \(\mathcal{S}=\{S_j\}\)，并在指定的 Universe（可覆盖对象集合）上选出预算内的子图集合 \(\mathcal{S}^\*\)，使覆盖最大且成本可控。

### 2.2 技术路线
1. **受控 LLM 三元组抽取**：严格限定实体类型与关系类型，减少漂移与不可解释输出（见 `src/kg_consturcution/prompt/triple_extraction_prompt.py:4-106`）。
2. **LLM 同义实体对齐**：分桶降低复杂度，并查集闭包保证等价关系传递闭包（见 `src/kg_consturcution/entity_alignment/stage4_clustering.py:10-54`）。
3. **图谱构建与证据锚点**：节点按 `(name,type)` 聚合，边携带 `source_row_index` 作为证据（见 `src/kg_consturcution/Step3_build_kg.py:66-134`）。
4. **报告诱导子图覆盖**：从种子节点扩展相关报告集合，再由报告诱导子图，语义合并后使用贪心集合覆盖选择（见 `src/defect_subgraph_cover/pipeline.py:51-142`）。

## 3. 算法细节

### 3.1 受控三元组抽取（Step1）
**核心思想**：将抽取空间限制为固定的实体类型与 7 种关系，要求“不推测、不扩展、不编造”，输出严格 JSON，从而让下游可以稳定解析并复用。

**关键实现**
- 批处理并发与失败重试：`src/kg_consturcution/Step1_triple_extractor.py:145-226`
- 输出清洗：`src/kg_consturcution/utils/sentence_utils.py:78-134`

**伪代码**
```text
Input: Excel reports with columns: id?, description
Output: kg_extraction_results.xlsx, triples.xlsx

items <- [(report_id, text)] from input where description not empty
results_map <- {}

parallel for batches(items):
  raw <- LLM(prompt(batch))
  json_text <- clean_llm_output(raw)
  parsed <- json.loads(json_text)
  emit per-item status + triples

retry up to MAX_RETRIES on failed items
save extraction results
flatten triples to triples.xlsx
```

### 3.2 同义实体对齐（Step2）
**核心思想**：实体同义会导致节点碎片化，进而影响覆盖与统计。对齐流程分四阶段：
1) 语义分桶；2) 可选桶间合并（多轮）；3) 桶内候选对生成与批量验证；4) 并查集闭包得到最终聚类。

**关键实现**
- 参数化入口：`src/kg_consturcution/Step2_synonym_alignment.py:28-75`
- 并查集闭包：`src/kg_consturcution/entity_alignment/stage4_clustering.py:10-54`

**伪代码**
```text
entities <- read_entities(triples.xlsx, type_filter)
buckets <- semantic_bucketing(entities, coarse_size, prefix_len)
if stage2_iterative:
  buckets <- merge_buckets(buckets, max_rounds)
pairs <- union over buckets: llm_validate_pairs(propose_pairs(bucket))
clusters <- union_find_closure(pairs, all_entities=entities)
write aligned_clusters.xlsx
```

### 3.3 图谱构建（Step3）
**核心思想**：用对齐聚类的“代表词”替换原三元组中的实体名，然后生成：
- `nodes.xlsx`：节点（含 `source_row_index`，表示其来源报告集合）
- `edges.xlsx`：边（含关系与来源报告集合）

**关键实现**
- 节点/边生成与证据聚合：`src/kg_consturcution/Step3_build_kg.py:66-155`

### 3.4 子图覆盖（defect_subgraph_cover）

#### 3.4.1 GraphIndex：图—报告双向索引
**核心思想**：将 `nodes.xlsx/edges.xlsx` 中的 `source_row_index` 解析为 `report_ids` 列表，建立 report 与 nodes/edges 的双向映射，并对边端点进行“report 关系回填”以增强闭包（使边出现的端点也能关联到报告）。

**关键实现**
- `split_report_ids()`：`src/defect_subgraph_cover/adapters/io.py:14-31`
- `build_graph_index()`：`src/defect_subgraph_cover/core/index.py:18-65`

#### 3.4.2 报告诱导扩展（Expansion）
**核心思想**：从种子节点出发，交替扩展报告集合 \(R\) 与节点集合 \(V\)：
- 初始 \(R \leftarrow reports(seed)\)，\(V \leftarrow \{seed\}\)
- 迭代 k 轮：由 \(R\) 找到相关节点并并入 \(V\)，再由 \(V\) 找到相关报告并并入 \(R\)
- 通过 `r_max` 与 `v_max` 控制增长；严格模式下超限会截断（保证稳定）

**关键实现**
- `expand_reports_from_seed()`：`src/defect_subgraph_cover/core/expand.py:18-101`

**伪代码**
```text
Input: seed_node, k, r_max, strict_r_max
R <- node_to_reports[seed_node]
V <- {seed_node}
for hop in 1..k:
  V <- V ∪ (⋃_{rid∈R} report_to_nodes[rid])
  R <- R ∪ (⋃_{nid∈V} node_to_reports[nid])
  if strict_r_max and |R| > r_max: truncate R by depth+sort; recompute V; break
  if soft_r_max and |R| >= r_max: break
return (R,V)
```

#### 3.4.3 子图构造、覆盖定义与成本
**核心思想**
- 子图由报告集合诱导：节点/边取这些报告关联的并集
- 覆盖集 `covered_universe` 定义为子图节点与 Universe 的交集
- 成本为线性组合：\(\text{cost}=\alpha|R|+\beta|V|+\gamma|E|\)

**关键实现**
- `make_subgraph()` 与 `compute_cost()`：`src/defect_subgraph_cover/core/subgraph.py:24-62`

#### 3.4.4 语义合并（Signature-based Weighted Jaccard）
**核心思想**：不同种子可能生成高度重叠的子图。合并通过“节点类型签名集合”衡量子图语义相似度，并按阈值合并，降低冗余。

实现要点：
- 将子图节点按类型映射到签名集合（ISSUE/PHEN/DIAG/OP/ELEM/MOD）
- 对不同签名集合计算加权 Jaccard，默认权重：ISSUE 0.45，PHEN 0.30，DIAG 0.20，OP 0.05
- 额外门控规则避免“无问题陈述”情况下误合并（见 `src/defect_subgraph_cover/core/merge.py:195-207`）

**关键实现**
- `merge_similar_subgraphs()`：`src/defect_subgraph_cover/core/merge.py:54-243`

#### 3.4.5 贪心集合覆盖（Greedy Set Cover）
**核心思想**：在预算 `budget` 下选择子图集合覆盖 Universe 中尽可能多的节点，采用贪心策略选取最大化“单位成本增益”的子图：
\[
\arg\max_{S} \frac{|cover(S)\setminus covered|}{cost(S)}
\]

**关键实现**
- `greedy_set_cover()`：`src/defect_subgraph_cover/core/set_cover.py:14-47`

## 4. 系统实现与复现指南

### 4.1 输入输出约定
- Step1 输入：Excel，至少包含 `description`；建议包含 `id`
  - 读取与过滤：`src/kg_consturcution/Step1_triple_extractor.py:149-162`
- Step1 输出：`kg_extraction_results.xlsx`、`triples.xlsx`
- Step2 输入：`triples.xlsx`，输出：`aligned_clusters.xlsx`
- Step3 输入：`triples.xlsx + aligned_clusters.xlsx`，输出：`graph_output/nodes.xlsx + edges.xlsx`
- 子图覆盖输入：`nodes.xlsx + edges.xlsx`，可选 `defects`（含 `id,description`）
  - 读取器：`src/defect_subgraph_cover/adapters/io.py:34-87`
- 子图覆盖输出：
  - Excel：`subgraph_cover_report.xlsx`（由 `cli.py` 控制）`src/defect_subgraph_cover/cli.py:208-238`
  - 每个子图的 `.json`（全量）与 `.brief.json`（摘要路径）`src/defect_subgraph_cover/adapters/export.py:35-89`

### 4.2 shared_output 的跨模块衔接
`defect_subgraph_cover` 默认从 `REPO_ROOT/shared_output` 读取 `nodes.xlsx/edges.xlsx`：
`src/defect_subgraph_cover/cli.py:34-43`。

## 5. 实验设置建议（论文体写法）

### 5.1 数据集与预处理
建议在论文中明确：
- 数据来源：缺陷报告的采集渠道、时间跨度、应用类型
- 数据规模：报告条数 \(N\)，平均字数、空文本比例
- 字段约定：`id` 是否存在、是否去重、是否包含噪声（如模板句/工单字段）

本代码假设至少存在 `description` 文本列；如缺失 `id` 则使用行号作为 report_id（见 `src/kg_consturcution/Step1_triple_extractor.py:156-162`）。

### 5.2 实体/关系体系
建议将 `triple_extraction_prompt.py` 中的实体定义与 7 种关系作为“标注体系/抽取体系”写入论文方法章节，并说明：
- 为什么需要受控（避免模型输出不可解析的类型/关系）
- `ISSUE/PHEN/DIAG/OP/MOD/ELEM` 的业务意义与可解释性

### 5.3 评价指标（建议至少三类）
1) **抽取质量**（Step1）
- 抽取成功率：`success / success_no_triples / failed` 分布（来自 `kg_extraction_results.xlsx`）
- 结构有效率：JSON 解析成功比例（间接反映 `clean_llm_output` 鲁棒性）
- 人工抽样准确率：随机抽样 \(m\) 条报告，统计实体/关系的精确率（可用双人复核）

2) **图谱质量**（Step2/Step3）
- 节点碎片化降低：对齐前后唯一实体数变化、聚类规模分布
- 证据密度：节点/边的平均 `report_ids` 数量
- 类型一致性：是否出现跨类型误合并（Step2 的 `--entity-type-filter` 可控制）

3) **覆盖效果**（defect_subgraph_cover）
- Universe 覆盖率：\(|covered|/|universe|\)（`cli.py` 会打印统计，见 `src/defect_subgraph_cover/cli.py:161-165`）
- 冗余度：合并前后子图数对比（`raw_subgraphs` vs `merged_subgraphs`）
- 可解释性：`brief.json` 的路径是否能代表故障链路（人工评估）

### 5.4 对比实验（可选）
建议与以下基线对比：
- 无同义对齐（跳过 Step2），直接构图 + 覆盖
- 无语义合并（将 `merge_jaccard` 设置为 1 或禁用合并逻辑）对覆盖结果冗余的影响
- 覆盖策略替换：仅按覆盖增益排序（不除以 cost）作为简化基线

## 6. 参数敏感性建议（可直接写入论文实验章节）

### 6.1 Step1（抽取阶段）
- `--batch-size`：过大可能导致输出长度超限或解析失败；过小会增加调用次数与成本（见 `src/kg_consturcution/Step1_triple_extractor.py:30-38`）。
- `LLM-config.yaml`：`temperature/max_tokens/concurrency/timeout/max_retries` 会影响稳定性与成本（加载逻辑见 `src/kg_consturcution/llm_client.py:10-47`）。

建议做敏感性：固定数据集，改变 batch_size（1/3/5）、temperature（0.0/0.2/0.5）比较失败率与每条报告平均 tokens。

### 6.2 Step2（对齐阶段）
关键参数（见 `src/kg_consturcution/Step2_synonym_alignment.py:40-75`）：
- `--coarse-size`：越大越省调用但更易混桶；越小更稳但成本高
- `--bucket-limit`：控制桶内候选对规模；过大增加验证成本、过小可能漏对齐
- `--batch-validation-size`：验证批大小；过大易失败，过小效率低
- `--prefix-len`：粗分桶键；对“选择X”类文本存在稳定性影响
- `--stage2-iterative/--stage2-max-rounds`：桶间合并强度；过强可能引入误合并

建议敏感性：在固定 triples.xlsx 上，调节 coarse-size（50/100/200）与 bucket-limit（100/150/250）比较聚类数、平均簇大小与人工误合并率。

### 6.3 Step3（构图阶段）
（见 `src/kg_consturcution/Step3_build_kg.py:1-23`）
- `--allowed-types`：Universe/下游覆盖的有效实体范围，直接决定图规模与覆盖对象
- `--node-id-prefix/--node-id-width`：ID 生成策略，影响与外部系统对接一致性

### 6.4 子图覆盖（defect_subgraph_cover）
关键参数（见 `src/defect_subgraph_cover/cli.py:68-83` 与 `pipeline.py:51-67`）：
- `--k`：扩展跳数；k 越大，子图更大、相似度更高、合并与覆盖更“粗”
- `--r_max` + `--soft_r_max`：控制每个 seed 的报告规模；严格模式更稳定但可能截断信息（见 `expand.py:55-96`）
- `--merge_jaccard`：语义合并阈值；越高合并越少，越低合并越多
- `--budget`：最终选中子图数量上限
- `--alpha/beta/gamma`：成本函数权重，控制对“报告数/节点数/边数”的惩罚

建议敏感性：网格搜索 \((k \in \{1,2,3\}, r\_max \in \{50,100,200\}, merge \in \{0.4,0.6,0.8\})\)，以覆盖率、选中子图数、平均子图规模为指标绘制曲线。

## 7. 复杂度分析（理论 + 工程视角）

### 7.1 Step1（LLM 抽取）
主要成本为 LLM 调用。若 batch_size 为 \(B\)，报告数为 \(N\)，则调用次数约为 \(\lceil N/B \rceil\)，总体耗时受并发 `concurrency` 与超时重试影响。解析与落盘为 \(O(N)\)。

### 7.2 Step2（同义对齐）
设实体数为 \(M\)，分桶后第 \(i\) 个桶大小为 \(m_i\)。
- 分桶：\(O(M)\)
- 桶内候选对枚举的最坏情况为 \(O(\sum_i m_i^2)\)，但实现通过 `bucket-limit` 控制上限
- 并查集闭包：近似 \(O(P \alpha(M))\)，其中 \(P\) 为等价对数量，\(\alpha\) 为反阿克曼函数（几乎常数）

工程上，LLM 验证对数与桶规模是主要成本来源，建议以 `bucket-limit` 与分桶策略控制。

### 7.3 Step3（构图）
三元组数为 \(T\)：
- 聚合节点：\(O(T)\)
- 生成边：\(O(T)\)
整体线性。

### 7.4 子图覆盖
记节点数 \(V\)、边数 \(E\)、报告数 \(R\)、种子数 \(S\)。
- 构建索引：遍历 nodes/edges 形成映射，约 \(O(V + E + \text{report_links})\)（见 `index.py:18-65`）
- 单个 seed 扩展：在 k 轮内反复遍历 \(R\) 与 \(V\) 的邻接映射，最坏 \(O(k(|R|+|V|))\)，并受 `r_max/v_max` 截断约束（见 `expand.py:55-99`）
- 合并：当前实现为排序后成对扫描式聚合，最坏接近 \(O(S^2)\)，但阈值过滤与 `max_nodes_after_merge` 可降低实际合并次数（见 `merge.py:149-227`）
- 贪心覆盖：每轮扫描剩余子图，预算为 \(B\)，子图数为 \(M\)，复杂度约 \(O(BM)\)（见 `set_cover.py:18-41`）

## 8. 局限性与风险
1. **LLM 抽取的可控性仍受模型漂移影响**：虽然提示词严格约束，但模型仍可能输出不规范 JSON，需要依赖 `clean_llm_output` 的鲁棒清洗（`sentence_utils.py:78-134`），极端情况下仍可能失败。
2. **同义对齐存在误合并风险**：尤其是跨类型/跨语境的相近短语。虽然提供了 `--entity-type-filter` 与分桶策略，但无法完全消除。
3. **证据锚点的粒度受输入数据影响**：`source_row_index` 作为 report_id 的集合，若原始 `id` 不稳定或数据混入多来源，会影响可解释性。
4. **覆盖目标的定义是启发式的**：Universe 默认只覆盖 `{用户感知现象, 系统诊断信息, 问题陈述}`（见 `pipeline.py:75-79`），该选择适用于“面向故障模式”的覆盖，但对“模块/操作”的覆盖能力取决于研究目标。
5. **合并与覆盖策略偏启发式**：signature 合并与贪心覆盖是可解释的，但不保证全局最优；当子图间相似结构复杂时可能出现局部最优陷阱。
6. **规模化瓶颈**：当报告、节点、子图数量显著增大时，合并（近 \(O(S^2)\)）与覆盖（\(O(BM)\)）会成为瓶颈，需要工程层面的剪枝或近似。

## 9. 可复现性建议（写入论文“Reproducibility”）
- 固定 `LLM-config.yaml` 中的模型与 temperature；记录每次运行的 prompt 与响应日志（Step1/Step2 已写入运行目录日志，见 `Step1_triple_extractor.py:34-72` 与 `Step2_synonym_alignment.py:22-27`）。
- 对抽取/对齐/覆盖分别保存中间产物（本项目已按 timestamp 写入目录）。
- 在论文中公开（或内部存档）：
  - 数据集版本与统计信息
  - 使用的实体/关系定义与提示词版本（`triple_extraction_prompt.py`）
  - 关键参数（batch-size/coarse-size/bucket-limit/k/r_max/merge_jaccard/budget/alpha/beta/gamma）

## 10. 附录：主入口与关键函数索引
- Step1 抽取入口：`src/kg_consturcution/Step1_triple_extractor.py:145-226`
- Step2 对齐入口：`src/kg_consturcution/Step2_synonym_alignment.py:280-289`
- Step3 构图入口：`src/kg_consturcution/Step3_build_kg.py:1-155`
- Step4 Neo4j 入口：`src/kg_consturcution/Step4_neo4j_importer.py:311-328`
- 子图覆盖 CLI：`src/defect_subgraph_cover/cli.py:61-240`
- 子图覆盖管道：`src/defect_subgraph_cover/pipeline.py:51-160`
- 合并算法：`src/defect_subgraph_cover/core/merge.py:54-243`
- 覆盖算法：`src/defect_subgraph_cover/core/set_cover.py:14-47`

