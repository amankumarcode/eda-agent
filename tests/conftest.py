import pandas as pd
import pytest
from unittest.mock import MagicMock

from core.dataframe_registry import clear as clear_registry
from core.dataframe_registry import register as register_df
from core.schema import AgentResult, EDARequest, OutputType
from core.state import AgentState


@pytest.fixture(autouse=True)
def reset_registry():
    """Clear the DataFrame registry before and after every test."""
    clear_registry()
    yield
    clear_registry()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age":    [25, 32, 47, 51, 23, 38, 61, 29, 44, 35],
        "income": [40000, 58000, 92000, 105000, 36000, 74000, 120000, 52000, 88000, 67000],
        "score":  [72, 85, 91, 78, 65, 88, 95, 70, 82, 79],
        "region": ["N", "S", "E", "W", "N", "E", "S", "W", "N", "E"],
    })


@pytest.fixture
def sample_request(sample_df: pd.DataFrame) -> EDARequest:
    session_id = "test-session-001"
    register_df(session_id, sample_df)
    return EDARequest(
        goal="Find key drivers of income",
        session_id=session_id,
        output_formats=[OutputType.REPORT, OutputType.JSON],
        metadata={
            "filename": "sample.csv",
            "shape": [10, 4],
            "columns": ["age", "income", "score", "region"],
        },
    )


@pytest.fixture
def base_state(sample_request: EDARequest) -> AgentState:
    return AgentState(
        request=sample_request,
        plan=None,
        plan_approved=False,
        agent_results=[],
        scored_results=[],
        rerun_agent=None,
        rerun_count=0,
        narrative=None,
        key_insights=[],
        caveats=[],
        output_paths={},
        messages=[],
        evaluation=None,
        evaluation_count=0,
        dispatched_agents=[],
        completed_agents=[],
        next_action=None,
    )


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Mocked LLM response")
    return llm
