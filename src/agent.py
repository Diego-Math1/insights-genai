import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_deepseek import ChatDeepSeek
from config.settings import settings

def get_jump_agent(df: pd.DataFrame):
    llm = ChatDeepSeek(
        model="deepseek-v4-pro",
        temperature=0.1,
        api_key=settings.DEEPSEEK_API_KEY,
        max_retries=2
    )
    
    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        # Usamos "openai-tools" porque é o agente mais moderno e compatível
        agent_type="openai-tools",
        allow_dangerous_code=True,
        prefix="""
        Você é o Cientista de Dados Sênior da Jump. 
        Você recebe um DataFrame com transações financeiras (incluindo flag_fraude).
        SUA MISSÃO: Responda perguntas de negócio com insights acionáveis. 
        Sempre que notar um padrão, sugira um modelo preditivo. 
        Se pedirem documentação, gere em Markdown com: Objetivo, Metodologia (citar Sklearn/PyTorch), Resultados e Próximos Passos (Azure/Databricks).
        Seja conciso, mas com alto valor técnico.
        """,
        handle_parsing_errors=True,
        max_iterations=5,   # evita loops infinitos
    )
