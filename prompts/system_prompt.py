"""Load and render checked-in SMIT retrieval-agent system prompt templates."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from config import HarnessConfig
from tools.schemas import final_answer_tool

PROMPTS_ROOT = Path(__file__).resolve().parent / "system"
AGENT_MAX_K_TEMPLATE = "agent_max_k.md"
AGENT_FIXED_K_TEMPLATE = "agent_fixed_k.md"


def system_prompt_root_for_version(version: str) -> Path:
    root = PROMPTS_ROOT / str(version).strip()
    if not root.is_dir():
        raise FileNotFoundError(f"Unknown system_prompt_version: {version!r}")
    return root.resolve()


@lru_cache(maxsize=32)
def _load_template(version: str, *, fixed_k: bool) -> str:
    root = PROMPTS_ROOT / str(version).strip()
    name = AGENT_FIXED_K_TEMPLATE if fixed_k else AGENT_MAX_K_TEMPLATE
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing system prompt template: {path} (fixed_k={fixed_k})"
        )
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def render_system_prompt(config: HarnessConfig) -> str:
    template = _load_template(config.system_prompt_version, fixed_k=config.uses_fixed_agent_k)
    rendered = (
        template.replace("{agent_k}", str(config.agent_prompt_k()))
        .replace("{max_search_calls}", str(config.max_search_calls))
    )
    if config.uses_search_only_strategy():
        return rendered
    final_name = final_answer_tool(config).name
    return rendered.replace("{final_answer_tool}", final_name)
