import os
import re
import requests
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 1. Tavily 工具（网页搜索）
load_dotenv()

tavily_tool = TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=5
)

# 2. OpenWeatherMap 工具（查天气）
@tool
def get_weather(city: str):
    """查询指定城市的实时天气（通过 OpenWeatherMap API）。"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "❌ 没有找到 OPENWEATHER_API_KEY，请先在环境变量里配置。"

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&lang=zh_cn&units=metric"
    resp = requests.get(url)
    if resp.status_code != 200:
        return f"❌ 查询天气失败: {resp.text}"
    data = resp.json()
    return {
        "城市": city,
        "天气": data["weather"][0]["description"],
        "温度": f"{data['main']['temp']}°C",
        "体感温度": f"{data['main']['feels_like']}°C",
        "湿度": f"{data['main']['humidity']}%"
    }

# 3. 注册工具
tools = [tavily_tool, get_weather]

# 4. 创建智能体
agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=tools,
    prompt=(
        "你是一个旅行规划助手。"
        "当用户需要实时天气时，必须调用 get_weather 工具；"
        "当用户需要旅游信息时，请调用 Tavily。"
    )
)

# 5. 初始化一个小模型来提取城市（兜底用）
llm_extractor = init_chat_model("openai:gpt-4o-mini")

def extract_city(user_input: str) -> str:
    """优先用正则匹配城市，匹配不到就用 LLM 提取"""
    # 简单的中文城市匹配
    match = re.search(r"(北京|上海|广州|深圳|东京|大阪|京都|横滨|纽约|巴黎|伦敦)", user_input)
    if match:
        return match.group(1)

    # 如果正则没找到 → LLM 提取
    msg = [{"role": "user", "content": f"请从以下文本中提取一个城市名称: {user_input}"}]
    response = llm_extractor.invoke(msg)
    return response.content.strip()

# 6. 包装一层，强制天气调用逻辑
def smart_agent(user_input: str):
    if "天气" in user_input:
        city = extract_city(user_input)
        weather = get_weather(city)
        print("🌤️ 实时天气:", weather)

        result = agent.invoke(
            {"messages": [
                {"role": "user", "content": f"{user_input} (城市: {city})"}
            ]}
        )
        print("🤖 Assistant:", result["messages"][-1].content)
    else:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]}
        )
        print("🤖 Assistant:", result["messages"][-1].content)

# 7. 测试
if __name__ == "__main__":
    query = "Today is 2025/8/25. Please help me plan a three-day trip to Tokyo this weekend and next Monday, and also check the weather in Tokyo on those days. Please display the planned dates and weather dates as well. Output should be in Chinese."
    smart_agent(query)
