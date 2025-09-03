#更加复杂的多Agent实例
from typing import TypedDict, List, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
import uuid
from datetime import datetime, date

# (1) 定义 State
class ResearchState(TypedDict):
    topic: str
    sub_questions: List[str]        # 研究子问题
    results: Dict[str, List[Dict]]  # 子问题: 搜索结果
    summaries: Dict[str, str]       # 子问题: 初步总结
    refined_summary: str            # 跨子问题的综合分析
    critique_result: str
    report: str
    memory_hits: List[str]
    current_time: str

# (2) 初始化
load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    collection_name="deep_research_memory",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
tavily = TavilySearchResults(max_results=6)

# (3) 各个角色 Agent

def add_time(state: ResearchState) -> ResearchState:
    state["current_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return state

def memory_retriever(state: ResearchState) -> ResearchState:
    query = state["topic"]
    docs = vectorstore.similarity_search(query, k=2)
    state["memory_hits"] = [d.page_content for d in docs] if docs else []
    return state

def planner(state: ResearchState) -> ResearchState:
    """拆解主题为子问题"""
    prompt = (
        f"主题: {state['topic']}\n\n"
        "请将这个主题拆解为3-5个关键子问题，覆盖现状、应用、技术、挑战、趋势。\n"
        "输出JSON数组，每个元素是一个子问题。"
    )
    resp = llm.invoke(prompt)
    import json
    try:
        sub_qs = json.loads(resp.content)
    except:
        sub_qs = [state['topic']]
    state["sub_questions"] = sub_qs
    return state

def researcher(state: ResearchState) -> ResearchState:
    """对每个子问题进行检索"""
    results = {}
    for sub_q in state.get("sub_questions", []):
        docs: List[Dict] = tavily.invoke(sub_q)
        results[sub_q] = docs or []
    state["results"] = results
    return state

def summarizer(state: ResearchState) -> ResearchState:
    """每个子问题生成要点"""
    summaries = {}
    for sub_q, docs in state.get("results", {}).items():
        if not docs:
            summaries[sub_q] = "未检索到资料。"
            continue
        joined = "\n\n".join(
            f"[{i+1}] {item.get('title')}\nURL: {item.get('url')}\n{item.get('content')}"
            for i, item in enumerate(docs[:6])
        )
        prompt = (
            f"子问题: {sub_q}\n"
            "请根据以下资料生成要点列表（<=8条），附引用编号 [1][2]。\n\n"
            f"{joined}\n\n要点："
        )
        resp = llm.invoke(prompt)
        summaries[sub_q] = resp.content
    state["summaries"] = summaries
    return state

def analyst(state: ResearchState) -> ResearchState:
    """跨子问题的综合分析"""
    all_summaries = "\n\n".join(
        f"【{k}】\n{v}" for k, v in state.get("summaries", {}).items()
    )
    prompt = (
        "你是分析员，请基于以下子问题总结进行整合：\n"
        "跨子问题提炼关键趋势、矛盾点和空白点\n"
        "输出一个综合总结，逻辑清晰，避免重复。\n\n"
        f"{all_summaries}\n\n"
        "请输出综合分析："
    )
    resp = llm.invoke(prompt)
    state["refined_summary"] = resp.content
    return state

def critic(state: ResearchState) -> ResearchState:
    prompt = (
        f"请检查以下综合总结是否完整、逻辑一致:\n{state['refined_summary']}\n\n"
        "仅回答 pass 或 retry。"
    )
    resp = llm.invoke(prompt)
    critique = resp.content.strip().lower()
    state["critique_result"] = "retry" if "retry" in critique else "pass"
    return state

def writer(state: ResearchState) -> ResearchState:
    all_sources = []
    for docs in state.get("results", {}).values():
        all_sources.extend([d.get("url") for d in docs if d.get("url")])
    sources_block = "\n".join(f"- {u}" for u in all_sources[:15])

    prompt = (
        f"主题: {state['topic']}\n\n"
        f"综合要点: {state['refined_summary']}\n\n"
        "请撰写研究报告（结构：摘要、现状与趋势、应用案例、挑战与风险、结论与展望），2000字左右。\n"
        "正文中引用保持 [1][2] 格式。\n"
        "最后加“参考来源”。"
    )
    resp = llm.invoke(prompt)
    state["report"] = resp.content + "\n\n参考来源：\n" + sources_block
    return state

def memory_saver(state: ResearchState) -> ResearchState:
    doc_id = str(uuid.uuid4())
    today = date.today().isoformat()
    content = f"【主题】{state['topic']} ({today})\n\n{state['refined_summary']}"
    vectorstore.add_texts([content], metadatas=[{"topic": state["topic"], "date": today}], ids=[doc_id])
    return state

# (4) 构建LangGraph
graph = StateGraph(ResearchState)

graph.add_node("time", add_time)
graph.add_node("memory_retriever", memory_retriever)
graph.add_node("planner", planner)
graph.add_node("researcher", researcher)
graph.add_node("summarizer", summarizer)
graph.add_node("analyst", analyst)
graph.add_node("critic", critic)
graph.add_node("writer", writer)
graph.add_node("memory_saver", memory_saver)

graph.set_entry_point("time")

graph.add_edge("time", "memory_retriever")
graph.add_edge("memory_retriever", "planner")
graph.add_edge("planner", "researcher")
graph.add_edge("researcher", "summarizer")
graph.add_edge("summarizer", "analyst")
graph.add_edge("analyst", "critic")
graph.add_conditional_edges("critic", lambda s: s["critique_result"], {"pass": "writer", "retry": "analyst"})
graph.add_edge("writer", "memory_saver")
graph.add_edge("memory_saver", END)

app = graph.compile()

# (5) 输出
if __name__ == "__main__":
    topic = "AI Agent在金融领域的最新应用（截至今日）"
    final_state = app.invoke({"topic": topic})

    print("\n研究报告\n")
    print(final_state["report"])

    # 保存到文件
    with open("research_report.txt", "w", encoding="utf-8", errors="ignore") as f:
        f.write(final_state.get("report", "（无报告内容）"))
