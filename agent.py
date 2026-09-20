from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from tools import get_pm25_data, predict_pm25, compute_stat, plot_trend

agent = create_agent(
    model=ChatOllama(model="qwen3:8b"),
    tools=[get_pm25_data, predict_pm25, compute_stat, plot_trend],
    system_prompt="Tu es un assistant spécialisé dans le reporting de qualité de l'air (PM2.5) pour Turin. Utilise les outils disponibles pour répondre avec des chiffres précis. Si une question sort de ton périmètre, dis-le clairement plutôt que d'inventer.",
)

if __name__ == "__main__":
    q = "Quelle était la concentration moyenne de PM2.5 en janvier 2025 ?"
    result = agent.invoke({"messages": [{"role": "user", "content": q}]})
    print(result["messages"][-1].content)
    