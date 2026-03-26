from langchain.schema import SystemMessage, HumanMessage

SYSTEM_CONCEPT = (
    "你是知识图谱实体归类专家。目标：对输入实体进行语义分桶，输出高泛化但精准的概念标签与定义。"
    "严格规则："
    "1) 分类边界：实体需判定所属类别：动作(点击/选择/输入/返回)、对象(日期/地点/科目/难度)、模块(页面/流程/设置)、资源(API/服务)、问题(缺失/错误/延迟/不合理)。"
    "2) 禁止跨类别合并：不同类别不得同桶，例如‘选择日期’(动作+对象)与‘练习模块’(模块)不得合并。"
    "3) 细化对象：‘选择X’类短语按对象维度分桶，如日期/地点/科目分别独立。"
    "4) 粒度一致：页面/窗口/弹窗需区分；流程与控件名不得合并。"
    "5) 输出严格JSON，字段：{concept, definition, scope_keywords, examples}。"
    "字段说明："
    "- concept：2-4字的概念标签，如‘日期选择’、‘考试地点’、‘用户登录’。"
    "- definition：一句话操作化定义，避免过泛描述。"
    "- scope_keywords：该桶的核心关键词数组，用于后续聚类检验。"
    "- examples：从输入中挑选2-4个代表项。"
)

SYSTEM_PAIRS = (
    "你是等价实体识别器。目标：在同概念实体中识别语义等价对并给出置信度。"
    "判定准则：类别必须一致；对象必须一致；介质差异(页面/窗口)或路由差异允许但降低分。"
    "评分Rubric：0.9=完全等价(类别与对象一致，仅表述差异)；0.7=可能等价(介质/路由/粒度略差异)；0.5=相关但不等价。"
    "输出严格JSON数组：每项为[实体A, 实体B, 置信度, 理由, category]。"
)

SYSTEM_BATCH_VALIDATOR = (
    "你是严格的QA审核员。任务：对批量候选等价对进行对抗式验证。"
    "若发现类别差异、对象不一致、介质差异(页面vs弹窗)、粒度不一致、流程与控件混合、语义范围不重叠，则输出‘否决’，并给出差异点数组。"
    "输出严格JSON对象，键为候选对ID，值为‘通过’或‘否决’。"
    "示例：{'1': '通过', '2': '否决'}"
)

def get_system_prompt_text() -> str:

    return SYSTEM_CONCEPT

def build_concept_messages_batch(items: list, type_hint: str = None, fewshot: dict = None):
    content = (f"类型提示：{type_hint}\n" if type_hint else "") + "请为以下多个分块生成概念与定义，并按ID输出JSON映射：\n"
    for it in items:
        pid = it.get("id")
        ents = it.get("entities")
        content += f"ID: {pid}\n实体列表：\n{__to_json(ents)}\n----------------\n"
    if fewshot:
        content += "\n示例：\n" + __to_json(fewshot)
    content += "\n请严格输出 JSON 对象，Key为ID，Value为包含{concept, definition, scope_keywords, examples}的对象。"
    return [SystemMessage(content=SYSTEM_CONCEPT), HumanMessage(content=content)]

def build_pairs_messages(concept, entities, type_hint: str = None, fewshot: list = None):

    human = (
        (f"类型提示：{type_hint}\n" if type_hint else "") +
        f"概念：{concept}\n实体：\n" + __to_json(entities) +
        ("\n示例：\n" + __to_json(fewshot) if fewshot else "")
    )

    return [SystemMessage(content=SYSTEM_PAIRS), HumanMessage(content=human)]

def build_batch_validator_messages(pairs_with_reasons: list):


    content = "请批量验证以下候选对：\n"
    for item in pairs_with_reasons:
        pid = item.get("id")
        pair = item.get("pair")
        reason = item.get("reason")
        content += f"ID: {pid}\n候选对: {pair[0]} vs {pair[1]}\n理由: {reason}\n----------------\n"

    content += "\n请严格输出 JSON 对象，Key为ID，Value为'通过'或'否决'。"
    return [SystemMessage(content=SYSTEM_BATCH_VALIDATOR), HumanMessage(content=content)]

def __to_json(obj) -> str:

    import json
    return json.dumps(obj, ensure_ascii=False)
