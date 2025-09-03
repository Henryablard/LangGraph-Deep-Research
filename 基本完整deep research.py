#基本完整deep research
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
import uuid, os, json
from datetime import date

#  初始化
load_dotenv()

#  Memory / Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def get_vectorstore(topic: str) -> Chroma:
    """安全初始化 vectorstore，支持多主题隔离"""
    name = f"deep_research_memory_{topic.replace(' ', '_')}"
    os.makedirs("./chroma_db", exist_ok=True)
    try:
        store = Chroma(
            collection_name=name,
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        if not getattr(store, "_collection", None):
            store.reset_collection()
    except Exception:
        store = Chroma(
            collection_name=name,
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        store.reset_collection()
    return store

#  LLM 多 Agent 分工
summarizer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
critic_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
writer_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

#  Search
tavily = TavilySearchResults(max_results=5)

#  数据结构
class QuestionNode:
    def __init__(self, text: str, parent=None):
        self.text: str = text
        self.parent: Optional['QuestionNode'] = parent
        self.children: List['QuestionNode'] = []
        self.summary: Optional[str] = None
        self.is_researched: bool = False

#  工具函数
def planner(topic: str) -> List[str]:
    prompt = (
        f"主题: {topic}\n\n"
        "请将这个主题拆解为3-5个关键子问题，覆盖现状、应用、技术、挑战、趋势。\n"
        "输出JSON数组，每个元素是一个子问题。"
    )
    resp = summarizer_llm.invoke(prompt)
    try:
        sub_qs = json.loads(resp.content)
    except:
        sub_qs = [topic]
    return sub_qs

def search_docs(query: str) -> List[Dict]:
    docs: List[Dict] = tavily.invoke(query)
    if not docs: return []
    return [d for d in docs if isinstance(d, dict)]

def summarize(query: str, docs: List[Dict]) -> str:
    joined = "\n\n".join(
        f"[{i+1}] {item.get('title','无标题')}\nURL: {item.get('url','无URL')}\n{item.get('content','')}"
        for i, item in enumerate(docs)
    )
    prompt = (
        f"子问题: {query}\n\n"
        "请总结以下资料为要点（<=6条），并保持引用 [1][2]：\n\n"
        f"{joined}\n\n要点："
    )
    resp = summarizer_llm.invoke(prompt)
    return getattr(resp, "content", str(resp))

def critic_summary(query: str, summary: str) -> bool:
    prompt = (
        f"子问题: {query}\n"
        f"以下是总结:\n{summary}\n\n"
        "请检查总结是否全面覆盖主要方面，是否存在矛盾或逻辑错误。\n"
        "回答 'OK' 表示合格，否则 'NOT OK'。"
    )
    resp = critic_llm.invoke(prompt)
    result = getattr(resp, "content", "NOT OK").strip().upper()
    return result == "OK"

def save_to_memory(vectorstore: Chroma, query: str, summary: str):
    doc_id = str(uuid.uuid4())
    today = date.today().isoformat()
    content = f"【查询】{query} ({today})\n\n{summary}"
    vectorstore.add_texts([content], metadatas=[{"query": query, "date": today}], ids=[doc_id])

def retrieve_memory(vectorstore: Chroma, query: str):
    docs = vectorstore.similarity_search(query, k=2)
    return docs

def write_report(topic: str, all_summaries: Dict[str,str]) -> str:
    joined = "\n\n".join(f"【{k}】\n{v}" for k,v in all_summaries.items())
    prompt = (
        f"主题: {topic}\n"
        f"各子问题总结:\n{joined}\n\n"
        "请撰写完整研究报告（2000字左右），正文保留引用 [1][2]。\n"
        "使用真实案例和数据支撑，结构化呈现。"
    )
    resp = writer_llm.invoke(prompt)
    return getattr(resp, "content", str(resp))

#  树与研究节点
def build_tree(root_text: str, sub_questions: List[str]) -> QuestionNode:
    root_node = QuestionNode(root_text)
    for sq in sub_questions:
        root_node.children.append(QuestionNode(sq, parent=root_node))
    return root_node

def add_children_from_summary(node: QuestionNode):
    if node.children or not node.summary: return
    lines = [line.strip() for line in node.summary.split("\n") if line.strip()]
    for line in lines:
        if line[0].isdigit():
            node.children.append(QuestionNode(line, parent=node))

def research_node(vectorstore: Chroma, node: QuestionNode):
    if node.is_researched: return
    mem_docs = retrieve_memory(vectorstore, node.text)
    if mem_docs:
        node.summary = "\n".join(d.page_content for d in mem_docs)
    else:
        docs = search_docs(node.text)
        if docs:
            node.summary = summarize(node.text, docs)
        else:
            node.summary = "未找到资料"
        # Critic 检查循环
        while not critic_summary(node.text, node.summary):
            docs = search_docs(node.text)
            node.summary = summarize(node.text, docs)
        save_to_memory(vectorstore, node.text, node.summary)
    node.is_researched = True
    add_children_from_summary(node)

#  菜单交互
def show_menu(node: QuestionNode, is_root=False) -> Dict:
    menu_text = f"\n📍 当前菜单: {node.text}\n"
    option_mapping = {}
    # 仅子节点编号
    for idx, child in enumerate(node.children, 1):
        preview = child.summary.split("\n")[0] if child.summary else "❌ 未研究"
        menu_text += f"{idx}. {child.text} | 🔎 {preview}\n"
        option_mapping[idx] = ("child", child)
    # 额外选项编号
    extra_options = []
    if is_root:
        extra_options = [
            ("添加自定义问题", "custom"),
            ("生成最终报告", "generate_report"),
            ("退出", "exit")
        ]
    else:
        extra_options = [("返回上一级", "back")]

    for j, (label, action) in enumerate(extra_options, len(node.children) + 1):
        menu_text += f"{j}. {label}\n"
        option_mapping[j] = (action, None)

    return menu_text, option_mapping

def collect_summaries(node: QuestionNode) -> Dict[str,str]:
    summaries = {}
    def _collect(n: QuestionNode):
        if n.summary:
            summaries[n.text] = n.summary
        for c in n.children:
            _collect(c)
    _collect(node)
    return summaries

#  主循环
if __name__ == "__main__":
    print("=== Deep Research Agent ===")
    topic = input("请输入研究主题: (英文) ").strip()
    vectorstore = get_vectorstore(topic)
    sub_questions = planner(topic)
    root_node = build_tree(topic, sub_questions)
    stack: List[QuestionNode] = [root_node]

    while stack:
        current_node = stack[-1]
        is_root = (current_node.parent is None)
        menu_text, option_mapping = show_menu(current_node, is_root=is_root)
        choice = input(menu_text + "请输入编号: ").strip()
        if not choice.isdigit():
            print("⚠️ 请输入数字编号。")
            continue
        choice = int(choice)
        if choice not in option_mapping:
            print("⚠️ 无效选择。")
            continue
        action, param = option_mapping[choice]

        if action == "child":
            node = param
            research_node(vectorstore, node)
            if node.children:
                stack.append(node)
            else:
                print(f"\n📌 {node.text} 的总结：\n{node.summary}\n")
        elif action == "back":
            stack.pop()
        elif action == "custom":
            q = input("请输入自定义问题: ").strip()
            new_node = QuestionNode(q, parent=current_node)
            current_node.children.append(new_node)
            research_node(vectorstore, new_node)
        elif action == "generate_report":
            summaries = collect_summaries(root_node)
            report = write_report(topic, summaries)
            print("\n📘 最终研究报告：\n")
            print(report)
            with open("deep_research_report.txt","w", encoding="utf-8") as f:
                f.write(report)
            print("✅ 已保存为 deep_research_report.txt")
        elif action == "exit":
            print("👋 已退出。")
            break
