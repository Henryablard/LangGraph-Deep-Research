from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# 定义一个状态机节点函数
def research_agent(state):
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke("给我总结一下量子计算的应用场景。")
    return {"result": response.content}

# 构建 LangGraph
graph = StateGraph(dict)
graph.add_node("research", research_agent)
graph.set_entry_point("research")
graph.add_edge("research", END)

# 编译
app = graph.compile()

# 运行
result = app.invoke({})
print(result)
