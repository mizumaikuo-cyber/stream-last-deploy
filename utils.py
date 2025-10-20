"""
Core utilities for LLM response generation and error formatting.
Clean, ASCII-safe, and compatible with LangChain 0.3.x.
"""

from __future__ import annotations

from dotenv import load_dotenv
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain

import constants as ct


# Load environment variables if present
load_dotenv()


def get_source_icon(source: str) -> str:
    if isinstance(source, str) and source.startswith("http"):
        return ct.LINK_SOURCE_ICON
    return ct.DOC_SOURCE_ICON


def build_error_message(message: str) -> str:
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def get_llm_response(chat_message: str):
    """Generate LLM response with retrieval augmentation.

    Returns a dict like {"answer": str, "context": List[Document], ...}
    so that UI can render both text and sources.
    """
    # Initialize LLM (LangChain 0.3 expects `model` param)
    llm = ChatOpenAI(model=ct.MODEL, temperature=ct.TEMPERATURE)

    # Optional: agent-based flow with max_iterations guard
    if getattr(ct, "USE_AGENT", False):
        agent_res = _try_agent_answer(chat_message, llm)
        if agent_res is not None:
            # Maintain chat history as LC messages (human + ai)
            st.session_state.chat_history = st.session_state.get("chat_history", [])
            st.session_state.chat_history.append(HumanMessage(content=chat_message))
            ans_text = agent_res.get("answer", "")
            if isinstance(ans_text, str):
                st.session_state.chat_history.append(AIMessage(content=ans_text))
            return agent_res

    # Prompt to create independent question based on history
    qg_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    qg_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qg_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Answer prompt depends on app mode
    qa_template = (
        ct.SYSTEM_PROMPT_DOC_SEARCH
        if st.session_state.get("mode") == ct.ANSWER_MODE_1
        else ct.SYSTEM_PROMPT_INQUIRY
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Build chain with retriever if available; fallback to LLM only
    retriever = st.session_state.get("retriever")
    if retriever is not None:
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, qg_prompt
        )
        doc_chain = create_stuff_documents_chain(llm, qa_prompt)
        chain = create_retrieval_chain(history_aware_retriever, doc_chain)
        # Retrieval chain will populate 'context' for doc_chain. Provide inputs expected by prompts.
        result = chain.invoke({
            "input": chat_message,
            "chat_history": st.session_state.get("chat_history", []),
        })
    else:
        # No retriever: answer with LLM only using qa_prompt
        # qa_prompt expects a 'context' variable; supply empty string when retriever is unavailable
        direct_chain = qa_prompt | llm
        answer_text = direct_chain.invoke({
            "input": chat_message,
            "chat_history": st.session_state.get("chat_history", []),
            "context": "",
        })
        # Normalize to expected dict shape
        normalized_answer = (
            answer_text.content if hasattr(answer_text, "content") else str(answer_text)
        )
        result = {"answer": normalized_answer, "context": []}

    # Maintain chat history as LangChain messages
    st.session_state.chat_history = st.session_state.get("chat_history", [])
    st.session_state.chat_history.append(HumanMessage(content=chat_message))
    answer_text = result.get("answer", "")
    if isinstance(answer_text, str):
        st.session_state.chat_history.append(AIMessage(content=answer_text))

    return result


def _try_agent_answer(chat_message: str, llm: ChatOpenAI):
    """Run an agent with a retrieval tool under a strict iteration cap.

    Returns result dict on success, or None to let callers fallback to normal flow.
    """
    try:
        from langchain.tools import Tool
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain_core.prompts import ChatPromptTemplate
    except Exception:
        return None

    retriever = st.session_state.get("retriever")
    if retriever is None:
        return None

    def search_tool_run(q: str) -> str:
        try:
            docs = retriever.get_relevant_documents(q)
        except Exception:
            return ""
        parts = []
        for d in docs[: int(getattr(ct, "RETRIEVAL_TOP_K", 5))]:
            txt = str(getattr(d, "page_content", ""))
            parts.append(txt[:500])
        return "\n\n".join(parts) if parts else ""

    tool = Tool(
        name="search_corpus",
        func=search_tool_run,
        description=(
            "社内の文書コーパスを検索するためのツールです。会社情報、部署、人事、株主優待、規程などの"
            "事実ベースの質問に使用してください。入力は日本語の簡潔なクエリで与えてください。"
        ),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたは社内情報特化型のアシスタントです。必要に応じて search_corpus ツールを使ってから、"
                "最終的な回答を日本語で出力してください。"
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    try:
        agent = create_react_agent(llm, [tool], prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=[tool],
            max_iterations=int(getattr(ct, "MAX_AGENT_ITERATIONS", 3)),
            early_stopping_method=getattr(ct, "AGENT_EARLY_STOP_METHOD", "generate"),
            verbose=False,
        )
        res = executor.invoke({"input": chat_message})
        output = res.get("output") or res.get("final_output") or ""
        return {"answer": output, "context": []}
    except Exception:
        return None