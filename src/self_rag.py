from typing import List, TypedDict, Literal, Annotated
import operator
import sqlite3
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from tavily import TavilyClient

from src.config import get_settings
from src.vectorstore import get_retriver


class RAGState(TypedDict, total = False):
    user_question: str
    question: str
    memory: Annotated[List[str], operator.add]
    retrieval_query: str
    web_query: str
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    support_status: Literal["fully_supported", "partially_supported", "no_support", ""]
    evidence: List[str]
    usefulness: Literal["useful", "not_useful", ""]
    use_reason: str
    support_retries: int
    retrieval_rewrites: int
    web_rewrites: int
    source_mode: Literal["internal", "web", "direct", "none"]
    used_web_search: bool
    trace: List[str]


class RetrieveDecision(BaseModel):
    should_retrieve: bool

class RelevanceDecision(BaseModel):
    is_relevant: bool

class SupportDecision(BaseModel):
    status: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)

class UsefulnessDecision(BaseModel):
    status: Literal["useful", "not_useful"]
    reason: str

class QueryRewrite(BaseModel):
    query: str


def _llm():
    s = get_settings()
    if not s.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    return ChatGoogleGenerativeAI(
        api_key = s.gemini_api_key,
        model = s.gemeni_model,
        temperature = 0
    )


def _trace(state : RAGState, item : str):
    return [*(state.get("trace") or []), item]


def _format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        if meta.get("source_type") == "web":
            head = f"[WEB {i}] {meta.get('title','')} | {meta.get('url','')}"
        else:
            head = f"[INTERNAL {i}] {meta.get('title') or meta.get('document_name') or meta.get('source','')}"
            if meta.get("page") is not None:
                head += f" | page {int(meta['page']) + 1}"
        blocks.append(f"{head}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)



def _memory_text(state : RAGState, limit : int = 4) -> str:
    items = state.get("memory") or []
    return "\n\n".join(items[-limit:]) if items else "No previous conversation context."


def contextualize_question(state : RAGState):
    history = _memory_text(state)
    user_question = state.get("user_question") or state.get("question", "")
    if not state.get("memory"):
        return {
            "question" : user_question,
            "trace" : _trace(state, "Memory: new incident session")
        }
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the newest user message as a standalone cloud-operations question using the previous conversation only when needed. Preserve service names, symptoms, errors, and constraints. If the message already stands alone, return it unchanged. Do not answer the question."),
        ("human", "Previous conversation:\n{history}\n\nNewest message:\n{question}"),
    ])

    out = _llm().with_structured_output(QueryRewrite).invoke(
        prompt.format_messages(history = history, question = user_question)
    )
    return {
        "question": out.query,
        "trace": _trace(state,
                        f"Memory contextualized question: {out.query}"
        )
    }



def commit_memory(state : RAGState):
    user_question = state.get("user_question") or state.get("question", "")
    answer = state.get("answer", "")
    route = state.get("source_node", "none")
    entry = f"User: {user_question}\nAssistant ({route}): {answer}"
    return {
        "memory": [entry],
        "trace": _trace(state, "SQLite memory checkpoint updated")
    }



def descide_retrival(state : RAGState):
    prompt = ChatPromptTemplate.from_template(
        ("system", "You decide whether the question needs retrieval. Choose true for cloud operations, production incidents, service runbooks, deployment procedures, infrastructure behavior, troubleshooting steps, specific/current technical facts, or whenever evidence is needed. Choose false only for generic technical explanations that can safely be answered from general knowledge. If unsure choose true."),
        ("human", "Question: {question}"),
    )
    out = _llm().with_structured_output(RetrieveDecision).invoke(prompt.format_messages(question = state["question"]))
    return {
        "need_retrival": out.should_retrive, 
        "trace": _trace(state, f"Retrival decision: {out.should_retrive}")
    }



def route_after_decide(state : RAGState) -> Literal["direct", "retrive"]:
    return "retrive" if state.get("need_retrival", True) else "direct"


def generate_direct(state : RAGState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer briefly from general technical knowledge only. Do not invent organization-specific infrastructure, runbooks, credentials, incident history, or deployment procedures."),
        ("human", "{question}"),
    ])
    ans = _llm().invoke(prompt.format_messages(question = state["question"])).content
    return {
        "awswer": ans,
        "source_node": "direct",
        "trace": _trace(state, "Generated direct answer") 
    }



def retrive_internal(state : RAGState):
    q = state.get("retrieval_query") or state["question"]
    docs = get_retriver().invoke(q)
    for d in docs:
        d.metadata = {
            **(d.metadata or {}),
            "source_type" : "internal"
        }
    return {
        "docs" : docs,
        "relevant_docs": [],
        "source_mode": "internal",
        "_trace": _trace(state, f"Internal retrival: {len(docs)} chunks")
    } 