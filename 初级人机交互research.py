#交互research
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
import uuid
from datetime import date

# 初始化
load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="interactive_search_memory",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
tavily = TavilySearchResults(max_results=5)

def search_and_display(query: str) -> List[Dict]:
    """调用 Tavily 搜索并打印结果"""
    docs: List[Dict] = tavily.invoke(query)
    if not docs:
        print("❌ 没有找到结果。")
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
        f"用户查询: {query}\n\n"
        "以下是相关资料，请总结为要点（<=6条），并保持引用 [1][2]：\n\n"
        f"{joined}\n\n要点："
    )
    resp = llm.invoke(prompt)
    print("\n📌 总结结果：\n")
    print(resp.content)
    return resp.content

def save_to_memory(query: str, summary: str):
    """存储到向量数据库"""
    doc_id = str(uuid.uuid4())
    today = date.today().isoformat()
    content = f"【查询】{query} ({today})\n\n{summary}"
    vectorstore.add_texts([content], metadatas=[{"query": query, "date": today}], ids=[doc_id])
    print("✅ 已保存到本地记忆库。")

def retrieve_memory(query: str):
    """从向量数据库中检索"""
    docs = vectorstore.similarity_search(query, k=2)
    if not docs:
        print("📂 记忆库中没有相关内容。")
    else:
        print("\n📂 记忆库检索结果：")
        for d in docs:
            print(d.page_content[:200], "...\n")

# 主交互循环
if __name__ == "__main__":
    print("=== 交互式搜索 Agent ===")
    print("输入主题进行搜索，输入 `exit` 退出。")

    while True:
        query = input("\n请输入搜索主题: ").strip()
        if query.lower() == "exit":
            break

        # 检索记忆
        retrieve_memory(query)

        # 搜索并展示
        docs = search_and_display(query)
        if not docs:
            continue

        # 是否总结
        do_sum = input("\n是否生成总结？(y/n): ").strip().lower()
        if do_sum == "y":
            summary = summarize(query, docs)

            # 保存
            do_save = input("\n是否保存到记忆库？(y/n): ").strip().lower()
            if do_save == "y":
                save_to_memory(query, summary)

    print("👋 已退出。")
