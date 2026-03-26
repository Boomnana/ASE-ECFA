# 缺陷子图覆盖（Defect Subgraph Cover）方法说明

## 1. 方法目标与问题定义

本模块实现一种“由缺陷报告诱导的子图覆盖”方法：给定一个知识图谱（节点/边）及其与缺陷报告的关联关系，从若干“种子节点”出发扩展得到候选子图集合，并在预定义的覆盖宇宙（universe）上执行集合覆盖（set cover）选择，得到一组在成本约束下具有高覆盖度的子图，用于后续分析、可视化或实验对比。

核心问题可形式化为：

- 给定覆盖宇宙 \(U\)（一组需要覆盖的“关键节点”ID），候选子图集合 \(\mathcal{S} = \{S_1,\dots,S_n\}\)，其中每个子图 \(S_i\) 对应一个可覆盖集合 \(C_i \subseteq U\)（模块中为 `covered_universe`），以及一个成本 \(cost_i\)。
- 目标是在预算 \(B\)（最多选择子图数量）下选择子集 \(\mathcal{S}^\* \subseteq \mathcal{S}\)，最大化覆盖 \(\left|\bigcup_{S_i \in \mathcal{S}^\*} C_i\right|\)，并倾向于较低成本（通过“收益/成本”比值贪心实现）。

本模块的流程入口为 [cli.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/cli.py) 与 [pipeline.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/pipeline.py)。

## 2. 输入数据与字段语义

### 2.1 输入表结构

模块默认以 Excel 作为输入：

- `nodes.xlsx`：节点表，必需列见 [adapters/io.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/adapters/io.py#L10-L45)
  - `id`：节点唯一标识（字符串化）
  - `name`：节点名称
  - `type`：节点类型（用于定义 seed/universe 与合并策略）
  - `source_row_index`：来源行索引（本方法将其解释为“与缺陷报告的关联 ID 列表”）
- `edges.xlsx`：边表，必需列见 [adapters/io.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/adapters/io.py#L48-L60)
  - `id`：边唯一标识
  - `source_id` / `target_id`：端点节点 ID
  - `relation`：关系类型
  - `source_type` / `target_type`：端点类型（用于输出与分析）
  - `source_row_index`：来源行索引（同样被解释为“与缺陷报告的关联 ID 列表”）
- `defects`（可选）：缺陷报告表（xlsx/csv/jsonl），需包含 `id, description` 列，用于把 report_id 映射为文本描述，加载逻辑见 [adapters/io.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/adapters/io.py#L63-L87)。

### 2.2 从 `source_row_index` 到 `report_ids` 的转换

节点/边与缺陷报告的关联由 `source_row_index` 字段承载。模块将其解析为 `report_ids: list[str]`：

- 分隔符：`| , ; 空白字符` 等（正则 `r"[|,;\s]+"`）
- 去空、去重、保序

实现见 [split_report_ids](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/adapters/io.py#L14-L31)，并在 [load_nodes_xlsx](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/adapters/io.py#L34-L45) / [load_edges_xlsx](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/adapters/io.py#L48-L60) 中映射到 `df["report_ids"]`。

该设计使得“结构化图谱数据”与“缺陷报告集合”之间形成可追溯的双向链接，为后续扩展与诱导提供基础。

## 3. 索引构建：GraphIndex 的闭包与映射

为支持高效的“从节点找报告 / 从报告找节点与边”，模块构造 `GraphIndex`（见 [core/index.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/index.py#L9-L66)），包含以下关键映射：

- `node_to_reports: Dict[node_id, Set[report_id]]`
- `report_to_nodes: Dict[report_id, Set[node_id]]`
- `report_to_edges: Dict[report_id, Set[edge_id]]`
- `node_id_to_row` 与 `edge_id_to_row`：用于输出与路径描述时回查 name/type/relation 等元信息
- `defect_id_to_text`：可选的 `report_id -> description`

值得注意的是索引构建包含一个“闭包补全（closure/backfill）”步骤：遍历 `edges` 的 `report_ids`，不仅将边归属到 `report_to_edges`，还会把边端点节点加入 `report_to_nodes`，并将报告反向补充到端点的 `node_to_reports`（见 [build_graph_index](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/index.py#L33-L49)）。其效果是：

- 即便某些端点节点在 `nodes.xlsx` 的 `source_row_index` 中未显式标注报告，也可通过边的关联被纳入报告-节点关系，从而提升扩展的连通性与召回率。

模块在 CLI 与报表中会统计该补全带来的影响（如 backfilled nodes/pairs），见 [cli.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/cli.py#L168-L176) 与 [reporting/excel.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/reporting/excel.py#L45-L53)。

## 4. Universe 与 Seed 的定义

### 4.1 Universe（覆盖宇宙）

`universe` 是覆盖目标节点集合 \(U\)，默认类型为：`{"用户感知现象", "系统诊断信息", "问题陈述"}`（见 [pipeline.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/pipeline.py#L75-L81)）。构建方式：

\[
U = \{ \text{id} \mid \text{nodes_df.type} \in \text{universe_types} \}
\]

实现为 [build_universe](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/pipeline.py#L23-L25)。

### 4.2 Seeds（种子节点集合）

种子用于生成候选子图，默认选取 `DEFAULT_UNIVERSE_TYPES` 里的所有类型（见 [pipeline.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/pipeline.py#L20-L21)），并通过 [pick_seeds](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/pipeline.py#L27-L28) 取出其 `id` 列表。

CLI 支持用 `--seed_types` 与 `--universe_types` 调参（见 [cli.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/cli.py#L79-L83)）。

## 5. 从种子到候选报告集合：k-hop 扩展

### 5.1 扩展状态

对每个种子节点 \(s\)，扩展过程维护：

- \(R\)：当前已收集的 report_id 集合
- \(V\)：当前已触达的 node_id 集合

初始：\(R_0 = node\_to\_reports[s]\)，\(V_0 = \{s\}\)（见 [expand.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/expand.py#L26-L30)）。

### 5.2 迭代规则（report ↔ node 交替闭包）

每一跳迭代：

1. 由报告扩展节点：\(V \leftarrow V \cup \bigcup_{r \in R} report\_to\_nodes[r]\)
2. 由节点扩展报告：\(R \leftarrow R \cup \bigcup_{v \in V} node\_to\_reports[v]\)

实现见 [expand_reports_from_seed](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/expand.py#L64-L92)。

### 5.3 规模约束与截断策略

为避免扩展爆炸，方法引入：

- `k`：最大迭代轮数
- `r_max`：报告集合大小上限
  - strict 模式：超过上限时执行截断（按“更浅层发现的报告优先”排序），并重算 \(V\)（见 [expand.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/expand.py#L44-L85)）
  - soft 模式：达到上限后提前停止，不截断（见 [expand.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/expand.py#L94-L96) 与 [cli.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/cli.py#L89-L90)）
- `v_max`：节点集合大小上限

扩展会返回停止原因（`stopped_reason`）用于实验与诊断（见 [ExpansionResult](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/expand.py#L9-L15)）。

## 6. 子图诱导：由报告集合生成结构子图

扩展得到的报告集合 \(R\) 被用于诱导子图 \(S\)：

- 节点集合：\(Nodes(S) = \bigcup_{r \in R} report\_to\_nodes[r]\)
- 边集合：\(Edges(S) = \bigcup_{r \in R} report\_to\_edges[r]\)

实现为 [induce_subgraph_from_reports](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/subgraph.py#L18-L31)。

覆盖集合定义为：

\[
C(S) = Nodes(S) \cap U
\]

对应字段 `covered_universe`，由 [make_subgraph](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/subgraph.py#L41-L63) 计算。

### 6.1 成本函数

模块使用线性加权成本：

\[
cost(S) = \alpha |R| + \beta |Nodes(S)| + \gamma |Edges(S)|
\]

实现见 [compute_cost](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/subgraph.py#L34-L39)，参数在 CLI 中可调（`--alpha/--beta/--gamma`）。

该成本同时鼓励覆盖更多报告（信息密度）并对结构复杂度施加惩罚，使后续集合覆盖具备“收益/结构复杂度”权衡。

## 7. 候选子图合并：去冗余与语义聚合

候选子图会执行合并以降低冗余、提高候选质量（入口见 [pipeline.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/pipeline.py#L131-L139)）。

合并实现为 [merge_similar_subgraphs](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/merge.py#L43-L244)，支持两种模式：

### 7.1 Report-Jaccard 模式（基于报告集合相似度）

- 相似度：\(J(A,B) = \frac{|R_A \cap R_B|}{|R_A \cup R_B|}\)
- 若 \(J \ge \tau\)（阈值 `thresh`），则将子图合并（union 报告/节点/边/覆盖集合），并重算成本
- 通过 `max_nodes_after_merge` 限制合并后的节点数上限，避免“巨图”吞并

实现见 [merge.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/merge.py#L70-L124) 与 [jaccard](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/merge.py#L7-L15)。

### 7.2 Signature 模式（基于类型签名的加权相似度）

当提供 `node_type_by_id` 时，默认进入 signature 模式（见 [merge.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/merge.py#L61-L68) 与 [pipeline.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/pipeline.py#L68-L73)）。

步骤：

1. 将节点按类型映射为签名分区（默认别名：问题陈述→ISSUE，感知现象→PHEN，诊断信息→DIAG，用户操作→OP 等），见 [_signature_sets](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/merge.py#L29-L41) 与 [merge.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/merge.py#L125-L149)。
2. 对分区集合计算加权 Jaccard：

\[
sim(A,B) = \frac{\sum_k w_k \cdot J(S^A_k, S^B_k)}{\sum_k w_k}
\]

实现见 [weighted_jaccard](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/merge.py#L18-L27)。
3. 引入启发式过滤以控制误合并：
   - ISSUE 几乎无交集时要求 PHEN∪DIAG 相似度足够高
   - ISSUE 有少量交集但不足时要求 PHEN/DIAG 提供补充证据

对应逻辑见 [merge.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/merge.py#L188-L214)。

合并后子图命名会附加 `_mN`，表示由 N 个原始子图聚合而来（见 [merge.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/merge.py#L117-L123) 与 [merge.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/merge.py#L232-L240)）。

## 8. 集合覆盖选择：Greedy Set Cover

在合并后的候选子图集合上执行贪心覆盖（见 [core/set_cover.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/core/set_cover.py#L14-L47)）。

在每一步，计算每个候选子图的新增覆盖：

\[
gain(S) = |C(S) \setminus Covered|
\]

并选择最大化收益/成本比：

\[
score(S) = \frac{gain(S)}{\max(cost(S), \epsilon)}
\]

其中 \(\epsilon=10^{-9}\) 防止除零。选择后更新 `covered`，并从候选集中移除已选子图；若达到预算或已完全覆盖 \(U\) 则停止。

该策略在大规模集合覆盖中具有良好效率，但不保证全局最优；其输出适用于实验中对覆盖率-子图数量曲线、成本-覆盖权衡等指标进行分析。

## 9. 输出与可复现实验材料

### 9.1 Excel 报告

模块会生成包含多张工作表的 `.xlsx` 报告，构建逻辑见 [build_excel_sheets](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/reporting/excel.py#L25-L228)：

- `selected`：最终选择的子图统计（reports/nodes/edges/covered_universe/cost）
- `sel_reports` / `sel_nodes` / `sel_edges`：选择子图所包含的报告、节点、边明细
- `seed_stats` / `skipped_seeds`：扩展阶段统计与跳过原因
- `uncovered_nodes` / `unreachable_nodes` / `missed_nodes`：覆盖诊断（分别对应未覆盖、候选无法触达、候选可触达但未被选中）
- `type_universe` / `type_covered` / `type_uncovered`：按类型的覆盖统计
- `meta`：索引闭包补全等元信息

输出写入由 [write_excel_report](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/reporting/excel.py#L231-L241) 实现。

### 9.2 子图 JSON 与摘要 brief.json

若在 CLI 中提供 `--brief_dir`，会在目录下以时间戳创建运行目录，并输出每个选中子图的两类 JSON（见 [cli.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/cli.py#L224-L238) 与 [export.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/adapters/export.py#L35-L90)）：

- `sg_xxxxx(_mN).json`：全量结构（reports/nodes/edges/cost/covered_universe/defects）
- `sg_xxxxx(_mN).brief.json`：摘要（defects + 关键路径 paths）

### 9.3 关键路径提取与压缩（brief 输出）

brief 输出的 `paths` 来自 [build_paths_descriptions](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/adapters/paths.py#L262-L342)，其流程为：

1. 在子图内部构建邻接表（仅保留子图内边），见 [_build_subgraph_adjacency](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/adapters/paths.py#L38-L57)
2. 选择起点集合（默认仅使用核心类型：问题陈述/感知现象/诊断信息/用户操作），见 [_pick_start_nodes](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/adapters/paths.py#L60-L66)
3. 以 BFS 方式枚举长度 \(\le L\) 的简单路径（不允许节点重复），并对路径进行打分排序、去除子路径冗余，见 [_extract_top_paths](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/adapters/paths.py#L93-L165)
4. 将大量相似路径压缩为 star / reverse-star / chain-star 等更适合人工阅读的描述格式，见 [_group_star_paths](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/adapters/paths.py#L168-L229) 与格式化函数（`_format_star/_format_chain...`）

该路径摘要用于快速呈现子图“代表性因果/流程线索”，适合在实验报告中做质性分析与案例展示。

## 10. 可调参数与实验建议

与实验直接相关的参数主要包括（入口见 [cli.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/cli.py#L61-L83) 与 [pipeline.py](file:///c:/Users/27826/Desktop/ECDA_v1.0/src/defect_subgraph_cover/pipeline.py#L51-L67)）：

- 扩展相关：`k`, `r_max`, `strict_r_max/soft_r_max`
  - 影响候选子图的规模、连通性与多样性
- 合并相关：`merge_jaccard`, `max_nodes_after_merge`
  - 影响冗余程度与候选质量
- 覆盖相关：`budget`, `alpha/beta/gamma`
  - 控制选择子图数量与结构复杂度惩罚强度，从而改变覆盖率曲线形态
- 目标定义：`seed_types`, `universe_types`
  - 改变“覆盖对象”的语义边界，是对比实验的重要变量

在科学研究设置下，建议将上述参数作为自变量，报告如下指标：候选覆盖上限（candidate coverage union/universe）、最终覆盖率、覆盖率-预算曲线、成本-覆盖权衡，以及未覆盖/不可达节点的类别分布与代表性案例路径。
