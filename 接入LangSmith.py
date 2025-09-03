#接入LangSmith
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

#  初始化环境
load_dotenv()
os.environ["LANGSMITH_PROJECT"] = "deep research tracing"  # 指定 LangSmith 项目
os.environ["LANGSMITH_TRACING"] = "true"

#  包装 OpenAI 客户端以支持 LangSmith 追踪
openai_client = wrap_openai(OpenAI())

@traceable
def llm_call(prompt: str, model: str="gpt-4o-mini", temperature: float=0.5) -> str:
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature
    )
    return resp.choices[0].message.content

#  Memory / Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def get_vectorstore(topic: str) -> Chroma:
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
        f"Topic: {topic}\n\n"
        "Please split this topic into 3-5 key sub-questions covering current status, applications, technologies, challenges, and trends.\n"
        "Output as a JSON array of sub-questions."
    )
    resp = llm_call(prompt)
    try:
        sub_qs = json.loads(resp)
    except:
        sub_qs = [topic]
    return sub_qs

def search_docs(query: str) -> List[Dict]:
    docs: List[Dict] = tavily.invoke(query)
    if not docs: return []
    return [d for d in docs if isinstance(d, dict)]

def summarize(query: str, docs: List[Dict]) -> str:
    joined = "\n\n".join(
        f"[{i+1}] {item.get('title','No title')}\nURL: {item.get('url','No URL')}\n{item.get('content','')}"
        for i, item in enumerate(docs)
    )
    prompt = (
        f"Sub-question: {query}\n\n"
        "Summarize the following documents into <=6 key points, keeping references [1][2]:\n\n"
        f"{joined}\n\nKey points:"
    )
    return llm_call(prompt)

def critic_summary(query: str, summary: str) -> bool:
    prompt = (
        f"Sub-question: {query}\n"
        f"Summary:\n{summary}\n\n"
        "Check whether the summary covers all major aspects and has no contradictions or logical errors.\n"
        "Answer 'OK' if acceptable, otherwise 'NOT OK'."
    )
    result = llm_call(prompt, temperature=0.0).strip().upper()
    return result == "OK"

def save_to_memory(vectorstore: Chroma, query: str, summary: str):
    doc_id = str(uuid.uuid4())
    today = date.today().isoformat()
    content = f"[Query] {query} ({today})\n\n{summary}"
    vectorstore.add_texts([content], metadatas=[{"query": query, "date": today}], ids=[doc_id])

def retrieve_memory(vectorstore: Chroma, query: str):
    docs = vectorstore.similarity_search(query, k=2)
    return docs

def write_report(topic: str, all_summaries: Dict[str,str]) -> str:
    joined = "\n\n".join(f"[{k}]\n{v}" for k,v in all_summaries.items())
    prompt = (
        f"Topic: {topic}\n"
        f"Sub-question summaries:\n{joined}\n\n"
        "Write a full research report (~2000 words), preserving references [1][2]. "
        "Use real cases and data, and present structured sections."
    )
    return llm_call(prompt, model="gpt-4o")

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
            node.summary = "No documents found."
        while not critic_summary(node.text, node.summary):
            docs = search_docs(node.text)
            node.summary = summarize(node.text, docs)
        save_to_memory(vectorstore, node.text, node.summary)
    node.is_researched = True
    add_children_from_summary(node)

#  菜单交互
def show_menu(node: QuestionNode, is_root=False) -> Dict:
    menu_text = f"\n📍 Current menu: {node.text}\n"
    option_mapping = {}
    for idx, child in enumerate(node.children, 1):
        preview = child.summary.split("\n")[0] if child.summary else "❌ Not researched"
        menu_text += f"{idx}. {child.text} | 🔎 {preview}\n"
        option_mapping[idx] = ("child", child)
    extra_options = []
    if is_root:
        extra_options = [
            ("Add custom question", "custom"),
            ("Generate final report", "generate_report"),
            ("Exit", "exit")
        ]
    else:
        extra_options = [("Back", "back")]
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
    topic = input("Enter research topic (English only): ").strip()
    vectorstore = get_vectorstore(topic)
    sub_questions = planner(topic)
    root_node = build_tree(topic, sub_questions)
    stack: List[QuestionNode] = [root_node]

    while stack:
        current_node = stack[-1]
        is_root = (current_node.parent is None)
        menu_text, option_mapping = show_menu(current_node, is_root=is_root)
        choice = input(menu_text + "Enter number: ").strip()
        if not choice.isdigit():
            print("⚠️ Please enter a number.")
            continue
        choice = int(choice)
        if choice not in option_mapping:
            print("⚠️ Invalid choice.")
            continue
        action, param = option_mapping[choice]

        if action == "child":
            node = param
            research_node(vectorstore, node)
            if node.children:
                stack.append(node)
            else:
                print(f"\n📌 {node.text} summary:\n{node.summary}\n")
        elif action == "back":
            stack.pop()
        elif action == "custom":
            q = input("Enter custom question: ").strip()
            new_node = QuestionNode(q, parent=current_node)
            current_node.children.append(new_node)
            research_node(vectorstore, new_node)
        elif action == "generate_report":
            summaries = collect_summaries(root_node)
            report = write_report(topic, summaries)
            print("\n📘 Final Research Report:\n")
            print(report)
            with open("deep_research_report.txt","w", encoding="utf-8") as f:
                f.write(report)
            print("✅ Saved as deep_research_report.txt")
        elif action == "exit":
            print("👋 Exited.")
            break
