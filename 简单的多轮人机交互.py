#多轮交互
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
import uuid
from datetime import date
import json

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

#  工具函数
def planner(topic: str) -> List[str]:
    """生成子问题"""
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
    """调用 Tavily 搜索并打印结果"""
    docs: List[Dict] = tavily.invoke(query)
    if not docs:
        print("没有找到结果。")
        return []
    for i, d in enumerate(docs, 1):
        print(f"\n[{i}] {d.get('title')}")
        print(f"URL: {d.get('url')}")
        print(f"{d.get('content')[:200]}...")  # 只显示前200字
    return docs

def summarize(query: str, docs: List[Dict]) -> str:
    """调用 LLM 总结"""
    joined = "\n\n".join(
        f"[{i+1}] {item.get('title')}\nURL: {item.get('url')}\n{item.get('content')}"
        for i, item in enumerate(docs)
    )
    prompt = (
        f"子问题: {query}\n\n"
        "以下是相关资料，请总结为要点（<=6条），并保持引用 [1][2]：\n\n"
        f"{joined}\n\n要点："
    )
    resp = llm.invoke(prompt)
    print("\n总结结果：\n")
    print(resp.content)
    return resp.content

def save_to_memory(query: str, summary: str):
    """存储到向量数据库"""
    doc_id = str(uuid.uuid4())
    today = date.today().isoformat()
    content = f"【查询】{query} ({today})\n\n{summary}"
    vectorstore.add_texts([content], metadatas=[{"query": query, "date": today}], ids=[doc_id])
    print("已保存到记忆库。")

def retrieve_memory(query: str):
    """从向量数据库中检索"""
    docs = vectorstore.similarity_search(query, k=2)
    if not docs:
        print("记忆库中没有相关内容。")
    else:
        print("\n记忆库检索结果：")
        for d in docs:
            print(d.page_content[:200], "...\n")

def final_report(topic: str, all_summaries: Dict[str, str]) -> str:
    """综合分析生成最终研究报告"""
    joined = "\n\n".join(f"【{k}】\n{v}" for k, v in all_summaries.items())
    prompt = (
        f"主题: {topic}\n\n"
        f"以下是各子问题总结：\n{joined}\n\n"
        "请撰写研究报告（结构就按照标准的研究报告来编排），2000字左右。\n"
        "正文中引用保持 [1][2] 格式。"
        "在撰写报告的时候一定要有真实有力的案例和数据做为支撑"
    )
    resp = llm.invoke(prompt)
    return resp.content

# 主交互流程
if __name__ == "__main__":
    print("多轮交互研究 Agent")
    topic = input("\n请输入研究主题: ").strip()
    sub_questions = planner(topic)
    print("\n系统生成的子问题：")
    for i, q in enumerate(sub_questions, 1):
        print(f"{i}. {q}")

    all_summaries = {}

# 初始化 all_questions 时包含原始子问题
all_questions = sub_questions.copy()  # 包含原始子问题

while True:
    menu_text = "\n请选择操作:\n"
    option_mapping = {}  # 编号 -> (类型, 参数)
    current_index = 1

    # 列出所有问题（原始 + 自定义）
    for q in all_questions:
        if q in all_summaries:
            # summary 前2行作为预览，拼成一行
            lines = all_summaries[q].split("\n")[:2]
            preview = " | ".join(line.strip() for line in lines)
            menu_text += f"{current_index}. {q} ✅ 已有总结: {preview}\n"
        else:
            menu_text += f"{current_index}. {q} ❌ 未研究\n"
        option_mapping[current_index] = ("question", q)
        current_index += 1

    # 额外功能
    extra_options = [
        ("添加自定义问题", "custom_question"),
        ("查看已研究的总结", "view_summaries"),
        ("生成最终报告", "generate_report"),
        ("退出", "exit")
    ]
    for label, action in extra_options:
        menu_text += f"{current_index}. {label}\n"
        option_mapping[current_index] = (action, None)
        current_index += 1

    # 用户输入
    choice = input(menu_text + "请输入编号: ").strip()
    if not choice.isdigit():
        print("⚠️ 请输入数字编号。")
        continue
    choice = int(choice)
    if choice not in option_mapping:
        print("⚠️ 编号无效，请重新输入。")
        continue

    action, param = option_mapping[choice]

    # 执行选择
    if action == "question":
        query = param
        retrieve_memory(query)
        docs = search_and_display(query)
        if docs:
            summary = summarize(query, docs)
            all_summaries[query] = summary
            save_to_memory(query, summary)

    elif action == "custom_question":
        query = input("请输入自定义问题: ").strip()
        all_questions.append(query)  # 自定义问题加入 all_questions
        retrieve_memory(query)
        docs = search_and_display(query)
        if docs:
            summary = summarize(query, docs)
            all_summaries[query] = summary
            save_to_memory(query, summary)

    elif action == "view_summaries":
        print("\n📂 已研究总结：")
        for q, s in all_summaries.items():
            print(f"\n--- {q} ---\n{s}\n")

    elif action == "generate_report":
        report = final_report(topic, all_summaries)
        print("\n📘 最终研究报告：\n")
        print(report)
        with open("interactive_research_report.txt", "w", encoding="utf-8", errors="ignore") as f:
            f.write(report)
        print("✅ 已保存为 interactive_research_report.txt")

    elif action == "exit":
        print("👋 已退出。")
        break


