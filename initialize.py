"""
Initialization utilities for Streamlit app.
Runs once on first page load to prepare logging, session, and retriever.
"""

from __future__ import annotations

import os
import sys
import logging
import unicodedata
from uuid import uuid4
from logging.handlers import TimedRotatingFileHandler

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import constants as ct


# Load .env early
load_dotenv()


def _ensure_openai_key() -> None:
    try:
        if not os.getenv("OPENAI_API_KEY"):
            key = None
            if hasattr(st, "secrets"):
                key = st.secrets.get("OPENAI_API_KEY", None)
            if key:
                os.environ["OPENAI_API_KEY"] = key
    except Exception:
        # Do not block app start if secrets is missing; downstream will show friendly error
        pass


def initialize() -> None:
    """Entry point to initialize session, logging and retriever."""
    _ensure_session_state()
    _ensure_openai_key()
    _initialize_logger()
    _initialize_retriever()


def _initialize_logger() -> None:
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    logger = logging.getLogger(ct.LOGGER_NAME)
    if logger.hasHandlers():
        return
    handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE), when="D", encoding="utf8"
    )
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.get('session_id','')}: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)


def _ensure_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex


def _initialize_retriever() -> None:
    if "retriever" in st.session_state:
        return

    try:
        docs_all = _load_data_sources()

        # Ingestion diagnostics: count PDFs and non-empty contents
        try:
            pdf_docs = [d for d in docs_all if str(d.metadata.get("source", "")).lower().endswith(".pdf")]
            pdf_nonempty = [d for d in pdf_docs if isinstance(d.page_content, str) and d.page_content.strip()]
            empty_examples = [str(d.metadata.get("source", "")) for d in pdf_docs if not (isinstance(d.page_content, str) and d.page_content.strip())][:5]
            st.session_state["ingest_stats"] = {
                "total_docs": len(docs_all),
                "pdf_count": len(pdf_docs),
                "pdf_nonempty_count": len(pdf_nonempty),
                "pdf_empty_examples": empty_examples,
            }
        except Exception:
            # Non-fatal
            pass

        # Normalize text on Windows hosts (defensive)
        if sys.platform.startswith("win"):
            for doc in docs_all:
                doc.page_content = _adjust_string(doc.page_content)
                for key in list(doc.metadata.keys()):
                    doc.metadata[key] = _adjust_string(doc.metadata[key])

        # If API key missing, avoid initializing embeddings and run without retriever
        if not os.getenv("OPENAI_API_KEY"):
            logging.getLogger(ct.LOGGER_NAME).warning(
                "OPENAI_API_KEY not found. Starting without retriever (LLM-only mode)."
            )
            st.session_state.retriever = None
            return

        embeddings = OpenAIEmbeddings()
        splitter = CharacterTextSplitter(
            chunk_size=int(getattr(ct, "CHUNK_SIZE", 500)),
            chunk_overlap=int(getattr(ct, "CHUNK_OVERLAP", 50)),
            separator="\n",
        )
        splitted_docs = splitter.split_documents(docs_all)

        db = Chroma.from_documents(splitted_docs, embedding=embeddings)
        st.session_state.retriever = db.as_retriever(
            search_kwargs={"k": int(getattr(ct, "RETRIEVAL_TOP_K", 5))}
        )
    except Exception as e:
        logging.getLogger(ct.LOGGER_NAME).warning(
            f"Retriever initialization failed; falling back to LLM-only mode. error={e}"
        )
        st.session_state.retriever = None


def _load_data_sources():
    docs_all = []
    _recursive_file_check(getattr(ct, "RAG_TOP_FOLDER_PATH", "./data"), docs_all)

    # Optional: load from web
    web_docs_all = []
    for web_url in getattr(ct, "WEB_URL_LOAD_TARGETS", []):
        try:
            loader = WebBaseLoader(web_url)
            web_docs = loader.load()
            web_docs_all.extend(web_docs)
        except Exception as e:
            logging.getLogger(ct.LOGGER_NAME).warning(
                f"WEB_URL_LOAD_TARGET skipped: {web_url} error={e}"
            )

    docs_all.extend(web_docs_all)
    return docs_all


def _recursive_file_check(path, docs_all) -> None:
    if os.path.isdir(path):
        for name in os.listdir(path):
            full_path = os.path.join(path, name)
            _recursive_file_check(full_path, docs_all)
    else:
        _file_load(path, docs_all)


def _file_load(path, docs_all) -> None:
    ext = os.path.splitext(path)[1]
    supported = getattr(ct, "SUPPORTED_EXTENSIONS", {})
    if ext in supported:
        try:
            loader = supported[ext](path)
            docs = loader.load()
            docs_all.extend(docs)
        except Exception as e:
            logging.getLogger(ct.LOGGER_NAME).warning(
                f"FILE_LOAD skipped: {path} error={e}"
            )


def _adjust_string(s):
    if not isinstance(s, str):
        return s
    if sys.platform.startswith("win"):
        s = unicodedata.normalize("NFC", s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    return s