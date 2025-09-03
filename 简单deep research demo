# deep research demo
# deep_research_minimal.py

from typing import TypedDict, List, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
import uuid
from datetime import datetime, date

# (1) 定义“有状态”的工作流状态
class ResearchState(TypedDict):
    topic: str                  # 研究主题（输入）
    results: List[Dict]         # 搜索原始结果（中间态）
    summary: str                # 聚合总结（中间态）
    critique_result: str        # 订正结果
    report: str                 # 最终报告（输出）
    memory_hits: List[str]      # 命中的历史知识
    current_time: str           # 时间戳

# (2) 初始化 LLM 与搜索工具
load_dotenv()  # 从 .env 里读取 OPENAI_API_KEY 和 TAVILY_API_KEY

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")          # 定义 Embeddings

vectorstore = Chroma(                               #构建向量储存
    collection_name="deep_research_memory",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

tavily = TavilySearchResults(max_results=6)        # 需要 TAVILY_API_KEY

# (3) 定义节点

def add_time(state: ResearchState) -> ResearchState:
    """获取当前日期时间"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state["current_time"] = now
    return state

def memory_retriever(state: ResearchState) -> ResearchState:
    """在正式搜索前，先查查数据库里有没有相关记忆"""
    query = state["topic"]
    docs = vectorstore.similarity_search(query, k=2)
    state["memory_hits"] = [d.page_content for d in docs] if docs else []
    return state

def researcher(state: ResearchState) -> ResearchState:
    """上网检索资料"""
    topic = state["topic"]
    docs: List[Dict] = tavily.invoke(topic)
    state["results"] = docs or []
    return state

def summarizer(state: ResearchState) -> ResearchState:
    """对检索结果做去重/压缩/提炼要点"""
    if not state.get("results"):
        state["summary"] = "未检索到足够的公开资料。"
        return state

    joined = "\n\n".join(
        f"[{i+1}] {item.get('title') or ''}\nURL: {item.get('url')}\n{item.get('content') or ''}"
        for i, item in enumerate(state["results"][:6])
    )

    prompt = (
        "你是研究助理。请从下列网页摘录中抽取关键事实，避免重复，保留时间线、机构名、数据点。\n"
        "要求：中文要点列表，不超过12条；若存在互相矛盾的信息要指出。\n"
        "请在每条要点后添加引用编号 [1][2]，编号对应下方资料顺序。\n\n"
        f"资料如下：\n{joined}\n\n"
        "请给出要点："
    )
    summary_msg = llm.invoke(prompt)
    state["summary"] = summary_msg.content if hasattr(summary_msg, "content") else str(summary_msg)
    return state

def critic(state: ResearchState) -> ResearchState:
    """检查 summary 是否合格，不合格就要求重试"""
    prompt = (
        "你是审稿人，请检查以下研究要点是否：\n"
        "1. 覆盖全面（没有明显遗漏主要信息）\n"
        "2. 逻辑自洽（没有明显矛盾）\n\n"
        f"研究要点：\n{state['summary']}\n\n"
        "回答仅输出：'pass' 或 'retry'"
    )
    critique_msg = llm.invoke(prompt)
    critique = critique_msg.content.strip().lower()
    state["critique_result"] = "retry" if "retry" in critique else "pass"
    return state

def writer(state: ResearchState) -> ResearchState:
    """把总结组织为结构化报告，并附参考来源"""
    urls = [r.get("url") for r in state.get("results", []) if r.get("url")]
    sources_block = "\n".join(f"- {u}" for u in urls[:10]) or "- （无来源链接）"

    prompt = (
      "请基于以下“研究要点”撰写一份结构化的中文研究报告，面向非专业读者：\n"
      "结构包含：标题、摘要、现状与趋势、代表性应用/案例、挑战与风险、结论与展望。\n"
      "尽量详细，2000字左右,不包含参考的字数，必须包含一些关键的数据点。\n\n"
      "每节使用标准 Markdown 格式(标题前面不要#号)，不要添加额外符号或说明。\n\n"
      "请一定注意报告的时态。一般提示词会提供时间和日期。\n"
      "在报告正文中请保留要点里的引用编号 [1][2]。\n"
      "报告末尾请添加“参考来源”部分，列出编号与对应网址。\n\n"
      f"【研究主题】{state['topic']}\n\n"
      f"【研究要点】\n{state['summary']}\n\n"
      "现在输出报告："
    )
    report_msg = llm.invoke(prompt)
    state["report"] = report_msg.content + "\n\n参考来源（自动收集，未逐条核验）：\n" + sources_block
    return state

def memory_saver(state: ResearchState) -> ResearchState:
    """把 summary 存进向量数据库"""
    doc_id = str(uuid.uuid4())
    today = date.today().isoformat()
    content = f"【主题】{state['topic']} （{today}）\n\n{state['summary']}"
    vectorstore.add_texts([content], metadatas=[{"topic": state["topic"], "date": today}], ids=[doc_id])
    return state

# (4) 构建LangGraph
graph = StateGraph(ResearchState)

graph.add_node("time", add_time)
graph.add_node("memory_retriever", memory_retriever)
graph.add_node("researcher", researcher)
graph.add_node("summarizer", summarizer)
graph.add_node("critic", critic)
graph.add_node("writer", writer)
graph.add_node("memory_saver", memory_saver)

#  只保留一个入口点
graph.set_entry_point("time")

#  顺序：time → memory_retriever → researcher → summarizer → critic → writer → memory_saver
graph.add_edge("time", "memory_retriever")
graph.add_edge("memory_retriever", "researcher")
graph.add_edge("researcher", "summarizer")
graph.add_edge("summarizer", "critic")
graph.add_conditional_edges("critic", lambda s: s["critique_result"], {"pass": "writer", "retry": "researcher"})
graph.add_edge("writer", "memory_saver")
graph.add_edge("memory_saver", END)

app = graph.compile()

# (5) 运行
if __name__ == "__main__":
    topic = "AI Agent在金融领域的最新应用（截至今日2025/8/25）"
    final_state = app.invoke({"topic": topic})

    print("\n研究报告\n")
    print(final_state["report"])

    # 保存到文件
    with open("research_report.txt", "w", encoding="utf-8", errors="ignore") as f:
        f.write(final_state.get("report", "（无报告内容）"))

