from langchain_ollama import ChatOllama

_models = {}

def get_llm(model_name: str = "i82blikeu/gemma-3n-E4B-it-GGUF:Q3_K_M"):
    global _models
    if model_name not in _models:
        _models[model_name] = ChatOllama(
            model=model_name,
            num_ctx=2048,
            num_predict=800,
            temperature=0.3,
            repeat_penalty=1.1,
        )
    return _models[model_name]