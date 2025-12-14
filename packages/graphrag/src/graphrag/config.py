from pydantic import BaseModel
import os

class Neo4jConfig(BaseModel):
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")

class LLMConfig(BaseModel):
    provider: str = os.getenv("LLM_PROVIDER", "hf_local")  # hf_local | openai | etc.
    model_name: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    device: str = os.getenv("LLM_DEVICE", "auto")
    max_new_tokens: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

class AppConfig(BaseModel):
    neo4j: Neo4jConfig = Neo4jConfig()
    llm: LLMConfig = LLMConfig()
