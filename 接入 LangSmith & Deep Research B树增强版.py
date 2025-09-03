# 接入 LangSmith & Deep Research B树增强版
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable
import uuid, os, json
from datetime import date
from concurrent.futures import ThreadPoolExecutor
import hashlib


# 初始化环境
load_dotenv()
os.environ["LANGSMITH_PROJECT"] = "deep_research_tracing"
os.environ["LANGSMITH_TRACING"] = "true"

# 包装 OpenAI 客户端以支持 LangSmith 追踪
openai_client = wrap_openai(OpenAI())

# LLM 调用
@traceable(name="LLM Call")
def llm_call(prompt: str, model: str="gpt-4o-mini", temperature: float=0.5) -> str:
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature
    )
    return resp.choices[0].message.content

# Memory / Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def get_vectorstore(topic: str) -> Chroma:
    topic_hash = hashlib.md5(topic.encode('utf-8')).hexdigest()
    name = f"deep_research_memory_{topic_hash}"
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

# Search
tavily = TavilySearchResults(max_results=5)

# ------------------ B 树节点 ------------------
class BTreeNode:
    def __init__(self, text: str, parent=None, max_children: int = 6):
        self.text: str = text
        self.parent: Optional['BTreeNode'] = parent
        self.children: List['BTreeNode'] = []
        self.summary: Optional[str] = None
        self.is_researched: bool = False
        self.max_children: int = max_children

    def add_child(self, child: 'BTreeNode'):
        if len(self.children) < self.max_children:
            self.children.append(child)
        else:
            # 超出 max_children，仍附加，B树可以再分裂扩展
            self.children.append(child)

# ------------------ 工具函数 ------------------
@traceable(name="Planner")
def planner(topic: str, n: int=6) -> List[str]:
    prompt = (
        f"主题: {topic}\n\n"
        f"请将这个主题拆解为{n}个关键子问题，覆盖现状、应用、技术、挑战、趋势。\n"
        "输出JSON数组，每个元素是一个子问题。不要回答无关内容。"
    )
    resp = llm_call(prompt)
    try:
        sub_qs = json.loads(resp)
    except:
        sub_qs = [topic]
    return sub_qs

@traceable(name="Search Docs")
def search_docs(query: str) -> List[Dict]:
    docs: List[Dict] = tavily.invoke(query)
    if not docs: return []
    return [d for d in docs if isinstance(d, dict)]

@traceable(name="Summarizer Agent")
def summarize(query: str, docs: List[Dict]) -> str:
    joined = "\n\n".join(
        f"[{i+1}] {item.get('title','无标题')}\nURL: {item.get('url','No URL')}\n{item.get('content','')}"
        for i, item in enumerate(docs)
    )
    prompt = (
        f"子问题: {query}\n\n"
        "请总结以下资料为要点（<=6条）：\n\n"
        f"{joined}\n\n要点："
    )
    return llm_call(prompt)

@traceable(name="Critic Agent")
def critic_summary(query: str, summary: str) -> bool:
    prompt = (
        f"子问题: {query}\n"
        f"以下是总结:\n{summary}\n\n"
        "请检查总结是否全面，是否覆盖主要方面，是否存在矛盾或逻辑错误。\n"
        "回答 'OK' 表示合格，否则 'NOT OK'。"
    )
    result = llm_call(prompt, temperature=0.0).strip().upper()
    return result == "OK"

def save_to_memory(vectorstore: Chroma, query: str, summary: str, topic: str):
    doc_id = str(uuid.uuid4())
    today = date.today().isoformat()
    content = f"[Query] {query} ({today})\n\n{summary}"
    vectorstore.add_texts([content], metadatas=[{"query": query, "date": today, "topic": topic}], ids=[doc_id])

def retrieve_memory(vectorstore: Chroma, query: str):
    docs = vectorstore.similarity_search(query, k=2)
    return docs

@traceable(name="Report Writer")
def write_report(topic: str, all_summaries: Dict[str,str]) -> str:
    joined = "\n\n".join(f"[{k}]\n{v}" for k,v in all_summaries.items())
    prompt = (
        f"主题: {topic}\n"
        f"各子问题总结:\n{joined}\n\n"
        "请撰写完整研究报告（2000字左右) "
        "报告要寻找一些可靠的参考，并且使用[1][2]标注出来 "
        "正文保留引用参考对应标号[1][2]。 "
        "使用真实案例和数据支撑，结构化呈现。 "
        "只引用提供的 sources。若无来源则明确标注“无来源”。"
    )
    return llm_call(prompt, model="gpt-4o")

# ------------------ B 树构建 ------------------
def build_btree(root_text: str, sub_questions: List[str], max_children: int = 6) -> BTreeNode:
    root_node = BTreeNode(root_text, max_children=max_children)
    for sq in sub_questions:
        root_node.add_child(BTreeNode(sq, parent=root_node, max_children=max_children))
    return root_node

def add_children_from_summary(node: BTreeNode):
    if node.children or not node.summary: return
    lines = [line.strip() for line in node.summary.split("\n") if line.strip()]
    for line in lines:
        if line[0].isdigit():
            node.add_child(BTreeNode(line, parent=node, max_children=node.max_children))

# ------------------ 研究节点 ------------------
MAX_RETRIES = 3

def research_node(vectorstore: Chroma, node: BTreeNode, topic: str):
    if node.is_researched: return
    mem_docs = retrieve_memory(vectorstore, node.text)
    if mem_docs:
        node.summary = "\n".join(d.page_content for d in mem_docs)
    else:
        docs = search_docs(node.text)
        node.summary = summarize(node.text, docs) if docs else "未找到资料。"
        retries = 0
        while not critic_summary(node.text, node.summary) and retries < MAX_RETRIES:
            docs = search_docs(node.text)
            node.summary = summarize(node.text, docs)
            retries += 1
        save_to_memory(vectorstore, node.text, node.summary, topic)
    node.is_researched = True
    add_children_from_summary(node)

def research_nodes_parallel(vectorstore: Chroma, nodes: List[BTreeNode], topic: str):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(research_node, vectorstore, node, topic) for node in nodes]
        for f in futures:
            f.result()

# ------------------ 菜单 ------------------
def show_btree_menu(node: BTreeNode, is_root=False) -> Dict:
    menu_text = f"\n📍 当前节点: {node.text}\n"
    option_mapping = {}
    for idx, child in enumerate(node.children, 1):
        preview = child.summary.split("\n")[0] if child.summary else "❌ Not researched"
        menu_text += f"{idx}. {child.text} | 🔎 {preview}\n"
        option_mapping[idx] = ("child", child)

    extra_options = []
    if is_root:
        extra_options = [
            ("添加自定义问题", "custom"),
            ("生成最终报告", "generate_report"),
            ("退出", "exit")
        ]
    else:
        extra_options = [("返回上一级", "back")]
    for j, (label, action) in enumerate(extra_options, len(node.children)+1):
        menu_text += f"{j}. {label}\n"
        option_mapping[j] = (action, None)
    return menu_text, option_mapping

def collect_summaries(node: BTreeNode) -> Dict[str,str]:
    summaries = {}
    def _collect(n: BTreeNode):
        if n.summary:
            summaries[n.text] = n.summary
        for c in n.children:
            _collect(c)
    _collect(node)
    return summaries

# ------------------ 主循环 ------------------
if __name__ == "__main__":
    print("    Deep Research Agent B-Tree (Enhanced)    ")
    topic = input("请输入研究主题: ").strip()
    n = input("请输入每级生成子问题数量 n: ").strip()
    n = int(n) if n.isdigit() and int(n)>0 else 6

    vectorstore = get_vectorstore(topic)
    sub_questions = planner(topic, n=n)
    root_node = build_btree(topic, sub_questions, max_children=n)

    # 并行研究所有子问题
    research_nodes_parallel(vectorstore, root_node.children, topic)

    # 菜单交互
    stack: List[BTreeNode] = [root_node]
    while stack:
        current_node = stack[-1]
        is_root = (current_node.parent is None)
        menu_text, option_mapping = show_btree_menu(current_node, is_root=is_root)
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
            research_node(vectorstore, node, topic)
            if node.children:
                stack.append(node)
            else:
                print(f"\n📌 {node.text} 的总结:\n{node.summary}\n")
        elif action == "back":
            stack.pop()
        elif action == "custom":
            q = input("请输入自定义问题: ").strip()
            new_node = BTreeNode(q, parent=current_node, max_children=current_node.max_children)
            current_node.add_child(new_node)
            research_node(vectorstore, new_node, topic)
        elif action == "generate_report":
            summaries = collect_summaries(root_node)
            report = write_report(topic, summaries)
            print("\n📘 最终研究报告:\n")
            print(report)
            with open("deep_research_report.txt","w", encoding="utf-8") as f:
                f.write(report)
            print("✅ 已保存为 deep_research_report.txt")
        elif action == "exit":
            print("👋 Exited.")
            break
