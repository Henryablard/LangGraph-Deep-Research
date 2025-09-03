# 调研报告问答助手(全链路清洗版)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document  # 用于重建干净的 Document
from dotenv import load_dotenv
import sys
import os

load_dotenv()

# 工具函数：彻底丢弃非法 surrogate 字符
def drop_surrogates(s: str) -> str:
    # 丢弃 U+D800–U+DFFF 范围内的代理码点（半个emoji等）
    return ''.join(ch for ch in s if not (0xD800 <= ord(ch) <= 0xDFFF))

def safe_str(x) -> str:
    return drop_surrogates(str(x))

def safe_write_text(path: str, text: str):
    # 写文件同样清洗并忽略无法编码的字符
    with open(path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(drop_surrogates(text))

# 让 Python 的标准输出在极端情况下也别因编码报错
os.environ["PYTHONIOENCODING"] = "utf-8"

# (1) 读取 PDF
pdf_path = r"C:\Users\ROG\OneDrive\Desktop\LangSmith调研报告.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# (2) 分块
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# (3) 在入库前清洗文本（从源头去掉非法字符，避免后续任意环节再出错）
clean_chunks = [
    Document(page_content=drop_surrogates(d.page_content), metadata=d.metadata)
    for d in chunks
]

# (4) 向量化 + 向量库
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(clean_chunks, embeddings)

# (5) 检索器
retriever = db.as_retriever(search_kwargs={"k": 3})         #从向量数据库里找到 最相关的 3 个文档块（chunk），然后把它们交给大模型参考，生成答案。

# (6) 问答链
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    retriever=retriever,
    return_source_documents=True
)

# (7) 提问
query = "这篇调研报告的主要内容有哪些？"
result = qa.invoke(query)

# (8) 输出（控制台与文件都走“安全字符串”）
print("\n答案:")
print(safe_str(result["result"]))

print("\n引用来源（仅展示前200字，完整内容写入 sources.txt）:")
lines_for_file = []
for i, doc in enumerate(result["source_documents"], 1):
    snippet = safe_str(doc.page_content)[:200]
    page = safe_str(doc.metadata.get("page", "未知"))
    print(f"\n来源 {i}: {snippet}...")
    print("页码:", page)

    # 也把完整段落写入文件
    full = safe_str(doc.page_content)
    lines_for_file.append(f"来源 {i}:\n{full}\n页码: {page}\n" + "="*80 + "\n")

safe_write_text("sources.txt", "\n".join(lines_for_file))     #在当前工作目录下面创建一个文件用来收录完整的引用内容
print("\n已保存完整引用到 sources.txt")
