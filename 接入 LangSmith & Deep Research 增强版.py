# 接入 LangSmith & Deep Research 增强版
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

import hashlib

def get_vectorstore(topic: str) -> Chroma:
    # 将 topic 转成哈希值，安全做 collection 名称
    topic_hash = hashlib.md5(topic.encode('utf-8')).hexdigest()
    name = f"deep_research_memory_{topic_hash}"  # 符合 Chroma 命名规范
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

# 数据结构
class QuestionNode:
    def __init__(self, text: str, parent=None):
        self.text: str = text
        self.parent: Optional['QuestionNode'] = parent
        self.children: List['QuestionNode'] = []
        self.summary: Optional[str] = None
        self.is_researched: bool = False

# 工具函数
@traceable(name="Planner")
def planner(topic: str) -> List[str]:
    prompt = (
        f"主题: {topic}\n\n"
        "请将这个主题拆解为3-5个关键子问题，覆盖现状、应用、技术、挑战、趋势。\n"
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
        "请撰写完整研究报告（2000字左右)"
        "报告要寻找一些可靠的参考，并且使用[1][2]标注出来"
        "正文保留引用参考对应标号[1][2]。\n"         #这里可以调整输出报告的字数
        "使用真实案例和数据支撑，结构化呈现。"
        "只引用提供的 sources。"
        "若无来源则明确标注“无来源”。"
    )
    return llm_call(prompt, model="gpt-4o")

# 树与研究节点
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

#  研究节点（并行化+重试限制）
MAX_RETRIES = 3

def research_node(vectorstore: Chroma, node: QuestionNode, topic: str):
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

def research_nodes_parallel(vectorstore: Chroma, nodes: List[QuestionNode], topic: str):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(research_node, vectorstore, node, topic) for node in nodes]
        for f in futures:
            f.result()

#  菜单
def show_menu(node: QuestionNode, is_root=False) -> Dict:
    menu_text = f"\n📍 当前菜单: {node.text}\n"
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

def collect_summaries(node: QuestionNode) -> Dict[str,str]:
    summaries = {}
    def _collect(n: QuestionNode):
        if n.summary:
            summaries[n.text] = n.summary
        for c in n.children:
            _collect(c)
    _collect(node)
    return summaries

# 主循环
if __name__ == "__main__":
    print("    Deep Research Agent (Enhanced)    ")
    topic = input("请输入研究主题: ").strip()
    vectorstore = get_vectorstore(topic)
    sub_questions = planner(topic)
    root_node = build_tree(topic, sub_questions)

    # 并行研究所有子问题
    research_nodes_parallel(vectorstore, root_node.children, topic)

    # 菜单交互
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
             # 如果该子问题还没有子问题，生成二级子问题
             if not node.children:
                 sub_questions = planner(node.text)  # 以该子问题为主题生成下一级子问题
                 for sq in sub_questions:
                     node.children.append(QuestionNode(sq, parent=node))
                 # 对二级子问题并行研究
                 research_nodes_parallel(vectorstore, node.children, topic)
             # 标记当前节点已研究
             research_node(vectorstore, node, topic)
             stack.append(node)

        elif action == "back":
            stack.pop()
        elif action == "custom":
            q = input("请输入自定义问题: ").strip()
            new_node = QuestionNode(q, parent=current_node)
            current_node.children.append(new_node)
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
