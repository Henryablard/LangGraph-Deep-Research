#多层级父子问题菜单
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults

# 初始化

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="interactive_research_memory",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
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

def planner(topic: str) -> List[str]:
    prompt = (
        f"主题: {topic}\n\n"
        "请将这个主题拆解为3-5个关键子问题，覆盖现状、应用、技术、挑战、趋势。\n"
        "输出JSON数组，每个元素是一个子问题。"
    )
    resp = llm.invoke(prompt)
    try:
        sub_qs = json.loads(resp.content)
    except:
        sub_qs = [topic]
    return sub_qs

def search_and_display(query: str) -> List[Dict]:
    docs: List[Dict] = tavily.invoke(query)
    if not docs:
        return []
    return [d for d in docs if isinstance(d, dict)]

def summarize(query: str, docs: List[Dict]) -> str:
    joined = "\n\n".join(
        f"[{i+1}] {item.get('title','无标题')}\nURL: {item.get('url','无URL')}\n{item.get('content','')}"
        for i, item in enumerate(docs)
    )
    prompt = (
        f"子问题: {query}\n\n"
        "以下是相关资料，请总结为要点（<=6条），并保持引用 [1][2]：\n\n"
        f"{joined}\n\n要点："
    )
    resp = llm.invoke(prompt)
    return getattr(resp, "content", str(resp))

def save_to_memory(query: str, summary: str):
    doc_id = str(uuid.uuid4())
    today = date.today().isoformat()
    content = f"【查询】{query} ({today})\n\n{summary}"
    vectorstore.add_texts([content], metadatas=[{"query": query, "date": today}], ids=[doc_id])

def retrieve_memory(query: str):
    docs = vectorstore.similarity_search(query, k=2)
    return docs

# 构建父子树

def build_tree(root_text: str, sub_questions: List[str]) -> QuestionNode:
    root_node = QuestionNode(root_text)
    for sq in sub_questions:
        root_node.children.append(QuestionNode(sq, parent=root_node))
    return root_node

def add_children_from_summary(node: QuestionNode):
    if node.children or not node.summary:
        return
    lines = [line.strip() for line in node.summary.split("\n") if line.strip()]
    for line in lines:
        if line[0].isdigit():
            node.children.append(QuestionNode(line, parent=node))

# 研究节点

def research_node(node: QuestionNode):
    if node.is_researched:
        return
    retrieve_memory(node.text)
    docs = search_and_display(node.text)
    if docs:
        node.summary = summarize(node.text, docs)
        save_to_memory(node.text, node.summary)
    else:
        node.summary = "未找到资料"
    node.is_researched = True
    add_children_from_summary(node)

# 菜单显示逻辑

def show_menu(node: QuestionNode, is_root=False) -> Dict:
    """返回菜单文本和编号映射"""
    menu_text = f"\n📍 当前菜单: {node.text}\n"
    option_mapping = {}
    for i, child in enumerate(node.children, 1):
        summary_preview = child.summary.split("\n")[0] if child.summary else "❌ 未研究"
        menu_text += f"{i}. {child.text} | 🔎 {summary_preview}\n"
        option_mapping[i] = ("child", child)
    idx_offset = len(node.children)

    # 根节点额外操作
    extra_options = []
    if is_root:
        extra_options = [
            ("添加自定义问题", "custom"),
            ("查看已研究总结", "view_summaries"),
            ("生成最终报告", "generate_report"),
            ("退出", "exit")
        ]
    else:
        extra_options = [("返回上一级", "back")]

    for j, (label, action) in enumerate(extra_options, idx_offset + 1):
        menu_text += f"{j}. {label}\n"
        option_mapping[j] = (action, None)

    return menu_text, option_mapping


# 收集所有总结

def collect_summaries(node: QuestionNode) -> Dict[str, str]:
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
    print("多轮交互研究 Agent")
    topic = input("请输入研究主题: ").strip()
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
            research_node(node)
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
            research_node(new_node)
        elif action == "view_summaries":
            summaries = collect_summaries(root_node)
            print("\n📂 已研究总结：")
            for k, v in summaries.items():
                print(f"\n--- {k} ---\n{v}\n")
        elif action == "generate_report":
            summaries = collect_summaries(root_node)
            report = final_report(topic, summaries)
            print("\n📘 最终研究报告：\n")
            print(report)
            with open("interactive_research_report.txt","w",encoding="utf-8") as f:
                f.write(report)
            print("✅ 已保存为 interactive_research_report.txt")
        elif action == "exit":
            print("👋 已退出。")
            break
