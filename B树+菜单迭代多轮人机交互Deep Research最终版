#B树+菜单迭代最终版
# coding: utf-8
import os
import uuid
import json
import hashlib
from typing import List, Dict, Optional
from datetime import date
from dotenv import load_dotenv

# 外部依赖（你已有的环境）
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults

# 初始化（请确保环境变量已配置）
load_dotenv()
os.environ.setdefault("LANGSMITH_PROJECT", "deep_research_tracing")
os.environ.setdefault("LANGSMITH_TRACING", "true")

# 包装 OpenAI（LangSmith 追踪）
openai_client = wrap_openai(OpenAI())

# embeddings / search
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
tavily = TavilySearchResults(max_results=5)

# 基本数据结构
class QuestionNode:
    def __init__(self, text: str, parent: Optional["QuestionNode"] = None):
        self.text: str = text
        self.parent: Optional["QuestionNode"] = parent
        self.children: List["QuestionNode"] = []
        self.summary: Optional[str] = None
        self.is_researched: bool = False

# LLM 封装
@traceable(name="LLM Call")
def llm_call(prompt: str, model: str="gpt-4o-mini", temperature: float=0.5) -> str:
    """
    简单封装 LLM 调用，返回字符串（content）
    """
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    # 兼容不同返回结构
    try:
        return resp.choices[0].message.content
    except Exception:
        # fallback
        return str(resp)

# Vectorstore 辅助（Chroma）
def get_vectorstore(topic: str) -> Chroma:
    topic_hash = hashlib.md5(topic.encode("utf-8")).hexdigest()
    name = f"deep_research_memory_{topic_hash}"
    os.makedirs("./chroma_db", exist_ok=True)
    try:
        store = Chroma(
            collection_name=name,
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        # 若需要重置等可在此处理
    except Exception:
        store = Chroma(
            collection_name=name,
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
    return store

def save_to_memory(vectorstore: Chroma, query: str, summary: str, topic: str):
    doc_id = str(uuid.uuid4())
    today = date.today().isoformat()
    content = f"[Query] {query} ({today})\n\n{summary}"
    try:
        vectorstore.add_texts([content], metadatas=[{"query": query, "date": today, "topic": topic}], ids=[doc_id])
    except Exception:
        pass

def retrieve_memory(vectorstore: Chroma, query: str):
    try:
        return vectorstore.similarity_search(query, k=2)
    except Exception:
        return []

#  Planner（生成 n 个子问题）
@traceable(name="Planner")
def planner(topic: str, n: int = 3) -> List[str]:
    """
    尝试让 LLM 返回 JSON 数组（严格），若失败则对行分割并清洗。
    返回去重后的子问题列表（最多 n 个）。
    """
    prompt = (
        f"主题: {topic}\n\n"
        f"请将这个主题拆解为 {n} 个关键子问题，覆盖现状、应用、技术、挑战、趋势。\n"
        "只返回 JSON 数组，示例： [\"子问题1\",\"子问题2\",...]\n"
        "不要输出其它说明。"
    )
    raw = llm_call(prompt, temperature=0.3)
    # 解析 JSON 优先
    items: List[str] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            items = [str(x).strip() for x in parsed]
    except Exception:
        # 退化为按行解析（去编号）
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        for ln in lines:
            # 去掉开头编号如 "1. " 或 "（1）"
            cleaned = ln.lstrip("0123456789.）) 、- ").strip()
            if cleaned:
                items.append(cleaned)
    # 清洗：去空、去重复、去与 topic 相同的项
    cleaned: List[str] = []
    seen = set()
    topic_lower = topic.strip().lower()
    for it in items:
        it_s = it.strip()
        if not it_s:
            continue
        key = it_s.lower()
        if key in seen:
            continue
        # 跳过与 topic 极度相似的结果
        if topic_lower in key or key in topic_lower:
            continue
        seen.add(key)
        cleaned.append(it_s)
        if len(cleaned) >= n:
            break

    # 如果 cleaned 不够，补齐常用模板（保证有 n 个）
    fallback_templates = [
        "现状与数据概览",
        "核心驱动因素/影响因素",
        "主要挑战与痛点",
        "典型案例与实践",
        "政策与法规影响",
        "未来趋势与展望",
        "技能/人才需求分析",
        "行业分布与机会"
    ]
    idx = 0
    while len(cleaned) < n:
        cand = f"{fallback_templates[idx % len(fallback_templates)]}"
        # 若重复则通过编号微调
        candidate = cand if cand not in cleaned else f"{cand} {len(cleaned)+1}"
        if candidate not in cleaned:
            cleaned.append(candidate)
        idx += 1

    return cleaned[:n]

# 搜索与摘要
@traceable(name="Search Docs")
def search_docs(query: str) -> List[Dict]:
    try:
        docs = tavily.invoke(query)
        return [d for d in docs if isinstance(d, dict)]
    except Exception:
        return []

@traceable(name="Summarizer Agent")
def summarize(query: str, docs: List[Dict]) -> str:
    # 将 docs 拼接成 prompt 给 LLM 做要点总结
    joined = "\n\n".join(
        f"[{i+1}] {item.get('title','无标题')}\nURL: {item.get('url','No URL')}\n{item.get('content','')}"
        for i, item in enumerate(docs)
    )
    prompt = (
        f"子问题: {query}\n\n"
        "请基于以下资料提炼 <=6 条要点（列出要点，每条不超过一行）：\n\n"
        f"{joined}\n\n要点："
    )
    resp = llm_call(prompt, temperature=0.2)
    return resp.strip()

@traceable(name="Critic Agent")
def critic_summary(query: str, summary: str) -> bool:
    prompt = (
        f"子问题: {query}\n"
        f"以下是总结:\n{summary}\n\n"
        "请检查总结是否全面，是否存在矛盾。只回答 OK 或 NOT OK。"
    )
    resp = llm_call(prompt, temperature=0.0).strip().upper()
    return resp == "OK"

# 节点展开（研究 + 生成下一层）
def expand_node(node: QuestionNode, n: int, vectorstore: Chroma, topic: str):
    """
    对 node 做 research 并生成 n 个子节点（下一层）。
    - 会先尝试从 memory 读取 summary
    - 若无，则检索并 summarize
    - 调用 planner 生成子问题，过滤重复/与父相同项；不足时补齐模板
    """
    if node.is_researched and node.children:
        return

    # research 当前节点（summary）
    mem_docs = retrieve_memory(vectorstore, node.text)
    if mem_docs:
        try:
            node.summary = "\n".join(d.page_content for d in mem_docs)
        except Exception:
            node.summary = str(mem_docs)
    else:
        docs = search_docs(node.text)
        if docs:
            node.summary = summarize(node.text, docs)
        else:
            # 如果没有搜到资料，标注为无来源
            node.summary = "未找到资料。"
        # 评审与重试（简单一次）
        try:
            if not critic_summary(node.text, node.summary):
                docs2 = search_docs(node.text)
                if docs2:
                    node.summary = summarize(node.text, docs2)
        except Exception:
            pass
        save_to_memory(vectorstore, node.text, node.summary, topic)

    node.is_researched = True

    # 生成下一层子问题（严格 n 个，且过滤掉与 node.text 相同或重复项）
    sub_qs = planner(node.text, n)
    # 过滤和去重
    filtered = []
    seen = set()
    node_lower = node.text.strip().lower()
    for s in sub_qs:
        s_str = s.strip()
        key = s_str.lower()
        if not s_str:
            continue
        if key in seen:
            continue
        # 跳过与父节点文本相同/高度重合的项
        if node_lower == key or node_lower in key or key in node_lower:
            continue
        seen.add(key)
        filtered.append(s_str)
    # 补齐（若 planner 产出被过滤后数量 < n）
    fallback_templates = [
        "现状与数据概览",
        "核心驱动因素/影响因素",
        "主要挑战与痛点",
        "典型案例与实践",
        "政策与法规影响",
        "未来趋势与展望",
        "技能/人才需求分析",
        "行业分布与机会"
    ]
    idx = 0
    while len(filtered) < n:
        candidate = f"{fallback_templates[idx % len(fallback_templates)]}"
        # 拼上父节点以避免过于通用导致歧义
        candidate_full = f"{node.text} — {candidate}"
        if candidate_full not in filtered:
            filtered.append(candidate_full)
        idx += 1

    # 创建子节点
    node.children = [QuestionNode(text=c, parent=node) for c in filtered[:n]]

# 递归预生成树（可选，保证进入时有下一层）
def research_tree(node: QuestionNode, n: int, vectorstore: Chroma, topic: str, max_depth: int = 2, current_depth: int = 0):
    expand_node(node, n, vectorstore, topic)
    if current_depth >= max_depth:
        return
    for child in node.children:
        research_tree(child, n, vectorstore, topic, max_depth, current_depth + 1)

# 收集 summaries
def collect_summaries(node: QuestionNode) -> Dict[str, str]:
    results: Dict[str, str] = {}
    def _collect(n: QuestionNode):
        if n.summary:
            results[n.text] = n.summary
        for c in n.children:
            _collect(c)
    _collect(node)
    return results

# 菜单显示（主菜单 / 节点菜单）
def show_menu(node: Optional[QuestionNode], is_main_menu: bool = False) -> (str, Dict[int, tuple]):
    """
    - 如果 is_main_menu=True：显示主菜单操作项
    - 否则：显示 node.children（下一层）
      若 node.children 为空：显示“查看总结 / 返回上一级”
    返回 (menu_text, option_mapping)
    option_mapping: {index: (action, param)}
    actions: "user_questions","custom_main","generate_report","exit","child","summary","back"
    """
    menu_lines: List[str] = []
    option_mapping: Dict[int, tuple] = {}
    idx = 1

    if is_main_menu:
        menu_lines.append("📍 当前菜单: 主菜单")
        menu_lines.append(f"{idx}. 用户问题"); option_mapping[idx] = ("user_questions", None); idx += 1
        menu_lines.append(f"{idx}. 新添加的问题"); option_mapping[idx] = ("custom_main", None); idx += 1
        menu_lines.append(f"{idx}. 写最终报告"); option_mapping[idx] = ("generate_report", None); idx += 1
        menu_lines.append(f"{idx}. 退出"); option_mapping[idx] = ("exit", None); idx += 1
    else:
        menu_lines.append(f"📍 当前菜单: {node.text}")
        if node.children:
            for child in node.children:
                menu_lines.append(f"{idx}. {child.text}")
                option_mapping[idx] = ("child", child)
                idx += 1
            # always allow return
            menu_lines.append(f"{idx}. 返回上一级"); option_mapping[idx] = ("back", None); idx += 1
        else:
            # 叶子节点：查看 summary 或 返回
            menu_lines.append(f"{idx}. 查看总结"); option_mapping[idx] = ("summary", node); idx += 1
            menu_lines.append(f"{idx}. 返回上一级"); option_mapping[idx] = ("back", None); idx += 1

    menu_text = "\n".join(menu_lines) + "\n"
    return menu_text, option_mapping

# 主程序
def main_loop():
    print("    Deep Research Agent    ")
    topic = input("请输入研究主题: ").strip()
    n_subs_in = input("请输入每级生成子问题数量 n（建议 2~6）: ").strip()
    n_subs = int(n_subs_in) if n_subs_in.isdigit() and int(n_subs_in) > 0 else 3
    max_depth_in = input("请输入树最大深度（建议 1~3）: ").strip()
    max_depth = int(max_depth_in) if max_depth_in.isdigit() and int(max_depth_in) >= 0 else 2

    vectorstore = get_vectorstore(topic)
    root_node = QuestionNode(topic)

    # 预生成树（确保进入「用户问题」时能直接显示第一级子问题）
    research_tree(root_node, n_subs, vectorstore, topic, max_depth=max_depth)

    stack: List[QuestionNode] = []
    in_main_menu = True

    while True:
        if in_main_menu:
            menu_text, option_mapping = show_menu(None, is_main_menu=True)
        else:
            current_node = stack[-1]
            # 如果某节点 children 还未生成（不应该出现），再确保生成一次
            if not current_node.children and current_node.is_researched:
                expand_node(current_node, n_subs, vectorstore, topic)
            menu_text, option_mapping = show_menu(current_node, is_main_menu=False)

        choice = input(menu_text + "请输入编号: ").strip()
        if not choice.isdigit():
            print("⚠️ 请输入数字编号。")
            continue
        idx = int(choice)
        if idx not in option_mapping:
            print("⚠️ 无效选择。")
            continue

        action, param = option_mapping[idx]

        # 主菜单操作
        if in_main_menu:
            if action == "user_questions":
                # 进入主题树（确保 root 已经展开）
                if not root_node.children:
                    expand_node(root_node, n_subs, vectorstore, topic)
                stack.append(root_node)
                in_main_menu = False
            elif action == "custom_main":
                q = input("请输入自定义问题（作为新根）：").strip()
                if not q:
                    print("⚠️ 问题为空，取消。")
                    continue
                new_node = QuestionNode(q, parent=None)
                expand_node(new_node, n_subs, vectorstore, topic)
                stack.append(new_node)
                in_main_menu = False
            elif action == "generate_report":
                all_summaries = collect_summaries(root_node)
                report = write_report(topic, all_summaries)
                print("\n📘 最终研究报告:\n")
                print(report)
                with open("deep_research_report.txt", "w", encoding="utf-8") as f:
                    f.write(report)
                print("✅ 已保存为 deep_research_report.txt")
            elif action == "exit":
                print("👋 退出。")
                break

        # 树节点菜单操作
        else:
            if action == "child":
                child_node: QuestionNode = param
                # 进入下一层：确保 child 的下一层已生成（如果你预生成了树，这里已经有 children）
                if not child_node.children and child_node.is_researched:
                    expand_node(child_node, n_subs, vectorstore, topic)
                stack.append(child_node)
            elif action == "summary":
                node: QuestionNode = param
                # 若尚未研究，先研究
                if not node.is_researched:
                    expand_node(node, n_subs, vectorstore, topic)
                print(f"\n📌 {node.text} 的总结:\n{node.summary}\n")
            elif action == "back":
                stack.pop()
                if len(stack) == 0:
                    in_main_menu = True

if __name__ == "__main__":
    main_loop()
