from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from tools import get_pm25_data, predict_pm25, compute_stat, plot_trend

agent = create_agent(
    model=ChatOllama(model="qwen3:8b"),
    tools=[get_pm25_data, predict_pm25, compute_stat, plot_trend],
    system_prompt=(
        "You are an assistant specialized in air quality (PM2.5) reporting for Turin. "
        "Use the available tools to answer with precise figures. "
        "IMPORTANT: 'predict_pm25' is ONLY for future dates (after today). For any past "
        "date or period, even if the question uses the word 'prediction', use "
        "'get_pm25_data' instead — that's real historical data, not a forecast. "
        "If a request is ambiguous between past and future, propose the relevant "
        "alternative rather than refusing outright. "
        "NEVER invent a causal explanation (weather, events...) that isn't explicitly "
        "returned by a tool. If a question is out of scope, say so clearly rather than "
        "making something up."
    ),
)

if __name__ == "__main__":
    q = input("Your question: ")
    result = agent.invoke({"messages": [{"role": "user", "content": q}]})
    print(result["messages"][-1].content)