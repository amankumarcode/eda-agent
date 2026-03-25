import os

from pydantic import BaseModel


class LLMConfig(BaseModel):
    model: str
    temperature: float

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            model=os.getenv("LLM_MODEL", "claude-sonnet-4-5"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        )


def get_llm_config() -> LLMConfig:
    return LLMConfig.from_env()


class AgentConfig(BaseModel):
    stat_analyst_use_react: bool
    evaluator_enabled: bool = True

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            stat_analyst_use_react=os.getenv(
                "STAT_ANALYST_USE_REACT", "false"
            ).lower() == "true",
            evaluator_enabled=os.getenv(
                "EVALUATOR_ENABLED", "true"
            ).lower() == "true",
        )


def get_agent_config() -> AgentConfig:
    return AgentConfig.from_env()


class AgentConfig(BaseModel):
    stat_analyst_use_react: bool
    evaluator_enabled: bool
    agents_required: list[str]  # new

    @classmethod
    def from_env(cls) -> "AgentConfig":
        required_raw = os.getenv("AGENTS_REQUIRED", "profiler,stat_analyst")
        return cls(
            stat_analyst_use_react=os.getenv(
                "STAT_ANALYST_USE_REACT", "false"
            ).lower() == "true",
            evaluator_enabled=os.getenv(
                "EVALUATOR_ENABLED", "true"
            ).lower() == "true",
            agents_required=[
                a.strip() for a in required_raw.split(",") if a.strip()
            ],
        )