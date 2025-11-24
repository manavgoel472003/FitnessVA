from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Sequence, Literal

import torch
from duckduckgo_search import DDGS
from dotenv import load_dotenv
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import ToolException
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, PrivateAttr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


@dataclass
class AgentConfig:
    model_id: str
    max_new_tokens: int = 512
    temperature: float = 0.5
    top_p: float = 0.9
    architecture_doc: Path = Path("fitness_va_architecture_design.md")
    exercise_db: Path = Path("exercisedb-api/src/data/exercises.json")
    user_profile: Path = Path("user_profile.json")


class ArchitectureDocArgs(BaseModel):
    question: str = Field(..., description="Specific architectural topic or question to look up.")


class ArchitectureDocTool(BaseTool):
    name: ClassVar[str] = "architecture_overview_reader"
    description: ClassVar[str] = (
        "Use this to consult the local fitness virtual assistant architecture design document "
        "for guidance on components, flows, tooling, and responsibilities. Provide a specific "
        "question or topic and this tool will return the most relevant sections."
    )
    args_schema: ClassVar[type[ArchitectureDocArgs]] = ArchitectureDocArgs
    _doc_path: Path = PrivateAttr()
    _sections: List[str] = PrivateAttr(default_factory=list)

    def __init__(self, doc_path: Path):
        super().__init__()
        self._doc_path = doc_path
        if not self._doc_path.exists():
            raise FileNotFoundError(f"Architecture document not found: {self._doc_path}")
        self._sections = self._load_sections()

    def _load_sections(self) -> List[str]:
        text = self._doc_path.read_text(encoding="utf-8")
        # Split on headings for lighter retrieval.
        chunks = re.split(r"\n(?=## )", text)
        cleaned = [chunk.strip() for chunk in chunks if chunk.strip()]
        return cleaned

    def _run(
        self,
        question: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        lower_question = question.lower()
        keywords = [w for w in re.findall(r"[a-z0-9]+", lower_question) if len(w) > 3]
        scores: List[tuple[int, str]] = []
        for idx, section in enumerate(self._sections):
            section_lower = section.lower()
            score = sum(section_lower.count(word) for word in keywords) if keywords else 0
            # Light bonus for earlier sections to keep context broad.
            scores.append((score + max(0, 5 - idx), section))
        top_sections = [section for _, section in sorted(scores, key=lambda item: item[0], reverse=True)[:3]]
        snippet = "\n\n".join(top_sections)
        if not snippet:
            snippet = self._sections[0]
        # Limit to avoid flooding prompts.
        return snippet[:2000]

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("ArchitectureDocTool does not support async execution.")


class ExerciseDatasetArgs(BaseModel):
    query: Optional[str] = Field(
        default=None,
        description="Free-text query to match against exercise names, instructions, or body parts.",
    )
    muscle: Optional[str] = Field(
        default=None,
        description="Optional primary muscle focus (e.g., chest, quads).",
    )
    equipment: Optional[str] = Field(
        default=None,
        description="Filter by available equipment (e.g., dumbbell, body weight).",
    )
    limit: int = Field(
        default=5,
        description="Maximum number of exercises to return (max 10).",
    )


class ExerciseDatasetTool(BaseTool):
    name: ClassVar[str] = "offline_exercise_dataset"
    description: ClassVar[str] = (
        "Use this to query the local ExerciseDB dataset that ships with the project. "
        "It can surface exercises filtered by muscles, equipment, or keywords and returns "
        "structured summaries you can cite in plans."
    )
    args_schema: ClassVar[type[ExerciseDatasetArgs]] = ExerciseDatasetArgs
    _dataset_path: Path = PrivateAttr()
    _dataset: List[Dict[str, Any]] = PrivateAttr(default_factory=list)

    def __init__(self, dataset_path: Path):
        super().__init__()
        self._dataset_path = dataset_path
        if not self._dataset_path.exists():
            raise FileNotFoundError(f"Exercise dataset not found: {self._dataset_path}")
        self._dataset = self._load_dataset()

    def _load_dataset(self) -> List[Dict[str, Any]]:
        with self._dataset_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _run(
        self,
        query: Optional[str] = None,
        muscle: Optional[str] = None,
        equipment: Optional[str] = None,
        limit: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        norm_query = (query or "").lower()
        muscle = (muscle or "").lower()
        equipment = (equipment or "").lower()
        limit = max(1, min(limit, 10))

        matches: List[Dict[str, Any]] = []
        for entry in self._dataset:
            if muscle and not any(muscle in m.lower() for m in entry.get("targetMuscles", [])):
                continue
            if equipment and not any(equipment in e.lower() for e in entry.get("equipments", [])):
                continue
            if norm_query:
                text = " ".join(
                    [
                        entry.get("name", ""),
                        entry.get("gifUrl", ""),
                        " ".join(entry.get("bodyParts", [])),
                        " ".join(entry.get("instructions", [])),
                    ]
                ).lower()
                if norm_query not in text:
                    continue
            matches.append(entry)
            if len(matches) >= limit:
                break

        if not matches:
            return (
                "No exercises matched those filters. Consider relaxing the query or checking spelling "
                "for muscle/equipment labels."
            )

        formatted = []
        for item in matches:
            formatted.append(
                "\n".join(
                    [
                        f"Name: {item.get('name')}",
                        f"Primary muscles: {', '.join(item.get('targetMuscles', []))}",
                        f"Body parts: {', '.join(item.get('bodyParts', []))}",
                        f"Equipment: {', '.join(item.get('equipments', []))}",
                        f"Instructions: {' '.join(item.get('instructions', [])[:4])}",
                        f"Exercise ID: {item.get('exerciseId')}",
                    ]
                )
            )

        return "\n\n---\n\n".join(formatted)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("ExerciseDatasetTool does not support async execution.")


class DuckDuckGoArgs(BaseModel):
    query: str = Field(..., description="Specific web search query to run.")
    max_results: int = Field(
        default=4,
        description="Maximum number of DuckDuckGo results to return (max 8).",
    )


class DuckDuckGoSearchTool(BaseTool):
    name: ClassVar[str] = "duckduckgo_web_search"
    description: ClassVar[str] = (
        "Use this tool for up-to-date web research via DuckDuckGo. Ideal for general guidelines, "
        "trend monitoring, or referencing reputable sources. Always cite the summarized sources."
    )
    args_schema: ClassVar[type[DuckDuckGoArgs]] = DuckDuckGoArgs

    def _run(
        self,
        query: str,
        max_results: int = 4,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        max_results = max(1, min(max_results, 8))
        normalized_query = query.strip()
        if not normalized_query:
            raise ToolException("DuckDuckGo search requires a non-empty query.")

        results: List[str] = []
        try:
            with DDGS() as ddgs:
                for item in ddgs.text(normalized_query, max_results=max_results):
                    title = item.get("title", "Untitled")
                    snippet = item.get("body", "")
                    href = item.get("href", "")
                    results.append(f"{title} — {snippet} ({href})")
        except Exception as exc:  # pragma: no cover - network errors
            raise ToolException(f"DuckDuckGo search failed: {exc}") from exc

        if not results:
            return "No DuckDuckGo results were returned."
        return "\n".join(results)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("DuckDuckGoSearchTool does not support async execution.")


class UserProfileArgs(BaseModel):
    operation: Literal["get", "update"] = Field(
        ...,
        description="Use 'get' to retrieve the current user profile. Use 'update' with profile fields to store them.",
    )
    profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Dictionary of user profile fields to upsert when operation is 'update'.",
    )


class UserProfileTool(BaseTool):
    name: ClassVar[str] = "user_profile_store"
    description: ClassVar[str] = (
        "Fetch or store information about the current user (age, gender, height_cm, weight_kg, goals, equipment, etc.). "
        "Always call this at the start of a conversation with operation='get'. If required fields are missing, "
        "politely ask the user for the missing values, then call the tool again with operation='update'. The data is saved locally."
    )
    args_schema: ClassVar[type[UserProfileArgs]] = UserProfileArgs
    _profile_path: Path = PrivateAttr()
    _required_fields: List[str] = PrivateAttr(default_factory=lambda: ["age", "gender", "height_cm", "weight_kg"])

    def __init__(self, profile_path: Path):
        super().__init__()
        self._profile_path = profile_path

    def _load_profile(self) -> Dict[str, Any]:
        if not self._profile_path.exists():
            return {}
        try:
            with self._profile_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
        except json.JSONDecodeError:
            pass
        return {}

    def _write_profile(self, profile: Dict[str, Any]) -> None:
        with self._profile_path.open("w", encoding="utf-8") as fh:
            json.dump(profile, fh, indent=2)

    def _format_missing(self, profile: Dict[str, Any]) -> List[str]:
        missing = []
        for field in self._required_fields:
            if not profile.get(field):
                missing.append(field)
        return missing

    def _run(
        self,
        operation: Literal["get", "update"],
        profile: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        current = self._load_profile()
        if operation == "get":
            if not current:
                missing = ", ".join(self._required_fields)
                return (
                    "NO_PROFILE: No stored user profile. Please ask the user for the following fields and call "
                    "user_profile_store again with operation='update': "
                    f"{missing}."
                )
            missing_fields = self._format_missing(current)
            if missing_fields:
                return (
                    "PARTIAL_PROFILE: The stored profile is missing the following fields: "
                    f"{', '.join(missing_fields)}. Current data: {json.dumps(current)}"
                )
            return f"PROFILE: {json.dumps(current)}"

        if operation == "update":
            if not profile:
                raise ToolException("operation 'update' requires the 'profile' argument.")
            current.update(profile)
            self._write_profile(current)
            missing_fields = self._format_missing(current)
            if missing_fields:
                return (
                    "PROFILE_UPDATED_PARTIAL: Saved data but still missing fields: "
                    f"{', '.join(missing_fields)}. Current profile: {json.dumps(current)}"
                )
            return f"PROFILE_UPDATED: {json.dumps(current)}"

        raise ToolException(f"Unsupported operation '{operation}' for user_profile_store.")

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("UserProfileTool does not support async execution.")


class ContextualSearchArgs(BaseModel):
    question: str = Field(..., description="The latest user request to contextualize.")
    focus: Optional[str] = Field(
        default=None,
        description="Optional hint about the target muscle/equipment focus if already parsed.",
    )


class ContextualSearchTool(BaseTool):
    name: ClassVar[str] = "contextual_search_pool"
    description: ClassVar[str] = (
        "Generate two tailored search queries (offline ExerciseDB + DuckDuckGo web) based on the user's latest request "
        "and stored profile, execute both searches, and return the combined context. "
        "Call this before finalizing any answer so responses cite both local and web data."
    )
    args_schema: ClassVar[type[ContextualSearchArgs]] = ContextualSearchArgs
    _dataset_path: Path = PrivateAttr()
    _profile_path: Path = PrivateAttr()
    _dataset: List[Dict[str, Any]] = PrivateAttr(default_factory=list)

    def __init__(self, dataset_path: Path, profile_path: Path):
        super().__init__()
        self._dataset_path = dataset_path
        self._profile_path = profile_path
        if not self._dataset_path.exists():
            raise FileNotFoundError(f"Exercise dataset not found: {self._dataset_path}")
        with self._dataset_path.open("r", encoding="utf-8") as fh:
            self._dataset = json.load(fh)

    def _load_profile(self) -> Dict[str, Any]:
        if not self._profile_path.exists():
            return {}
        try:
            with self._profile_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
        except json.JSONDecodeError:
            pass
        return {}

    def _extract_keywords(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        stop = {"what", "with", "that", "this", "have", "about", "from", "your", "please"}
        keywords = [tok for tok in tokens if len(tok) > 3 and tok not in stop]
        return keywords[:5]

    def _build_queries(
        self, question: str, profile: Dict[str, Any], focus: Optional[str]
    ) -> tuple[str, str]:
        keywords = self._extract_keywords(question)
        primary = focus or profile.get("primary_goal") or (keywords[0] if keywords else "fitness")
        equipment = ""
        equipment_list = profile.get("equipment_available") or []
        if isinstance(equipment_list, list) and equipment_list:
            equipment = equipment_list[0]
        elif isinstance(equipment_list, str) and equipment_list:
            equipment = equipment_list.split(",")[0]
        dataset_query = " ".join(part for part in [primary, equipment, "workout ideas"] if part)
        days = profile.get("preferred_training_days_per_week")
        web_parts = [primary, "training tips"]
        if days:
            web_parts.append(f"{days} days/week")
        web_query = " ".join(web_parts)
        return dataset_query, web_query

    def _search_dataset(self, query: str, limit: int = 4) -> List[Dict[str, Any]]:
        words = self._extract_keywords(query)
        scored: List[tuple[int, Dict[str, Any]]] = []
        for entry in self._dataset:
            haystack = " ".join(
                [
                    entry.get("name", ""),
                    " ".join(entry.get("targetMuscles", [])),
                    " ".join(entry.get("bodyParts", [])),
                    " ".join(entry.get("equipments", [])),
                    " ".join(entry.get("instructions", [])),
                ]
            ).lower()
            score = sum(1 for word in words if word in haystack)
            if score:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    def _search_web(self, query: str, max_results: int = 3) -> List[str]:
        results: List[str] = []
        try:
            with DDGS() as ddgs:
                for item in ddgs.text(query, max_results=max_results):
                    title = item.get("title", "Untitled")
                    snippet = item.get("body", "")
                    href = item.get("href", "")
                    results.append(f"{title} — {snippet} ({href})")
        except Exception as exc:
            results.append(f"DuckDuckGo error: {exc}")
        return results

    def _run(
        self,
        question: str,
        focus: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        profile = self._load_profile()
        dataset_query, web_query = self._build_queries(question, profile, focus)
        dataset_hits = self._search_dataset(dataset_query)
        web_hits = self._search_web(web_query)

        dataset_summary = []
        for item in dataset_hits:
            dataset_summary.append(
                f"{item.get('name')} — muscles: {', '.join(item.get('targetMuscles', []))}, "
                f"equipment: {', '.join(item.get('equipments', []))}, id: {item.get('exerciseId')}"
            )
        if not dataset_summary:
            dataset_summary.append("No offline exercises matched the generated query.")

        if not web_hits:
            web_hits = ["No DuckDuckGo results were returned."]

        profile_text = json.dumps(profile) if profile else "No stored profile."
        return "\n".join(
            [
                f"PROFILE_CONTEXT: {profile_text}",
                f"DATASET_QUERY: {dataset_query}",
                "DATASET_RESULTS:",
                *("  - " + line for line in dataset_summary),
                f"WEB_QUERY: {web_query}",
                "WEB_RESULTS:",
                *("  - " + line for line in web_hits),
            ]
        )

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("ContextualSearchTool does not support async execution.")


def _build_llm(config: AgentConfig) -> BaseLanguageModel:
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": "auto",
    }
    if hf_token:
        model_kwargs["token"] = hf_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    hf_pipeline = HuggingFacePipeline(pipeline=gen_pipeline)
    return ChatHuggingFace(llm=hf_pipeline)


def build_fitness_agent(config: AgentConfig):
    load_dotenv()
    llm = _build_llm(config)
    tools = [
        ArchitectureDocTool(config.architecture_doc),
        ExerciseDatasetTool(config.exercise_db),
        DuckDuckGoSearchTool(),
        UserProfileTool(config.user_profile),
        ContextualSearchTool(config.exercise_db, config.user_profile),
    ]
    instructions = (
        "You are a Personal Fitness Virtual Assistant focused on safe, structured coaching. "
        "Follow the architecture guidance from the architecture_overview_reader tool so your "
        "behavior aligns with the intended pipeline (AssistantAgent, LLMManager, tools, etc.). "
        "Favor the offline_exercise_dataset tool for exercise lookups. Use duckduckgo_web_search "
        "only for up-to-date general info or guidelines, and always mention the sources. "
        "Be explicit when a recommendation is based on local data vs web search. "
        "Always provide medical disclaimers when discussing pain, injuries, or health concerns. "
        "At the start of a conversation call user_profile_store with operation='get'; if required fields are missing, "
        "ask the user for age, gender, height_cm, and weight_kg (and any relevant goals/equipment) before proceeding. "
        "After understanding the request call contextual_search_pool with the user's latest message to automatically "
        "generate both offline ExerciseDB and DuckDuckGo search results, and weave that context plus the stored profile "
        "and conversation history into your final response."
    )
    return create_react_agent(model=llm, tools=tools, prompt=instructions)


def run_agent_once(agent, messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    result = agent.invoke({"messages": list(messages)})
    return result["messages"]
