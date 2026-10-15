
import os
import re
import uuid
import asyncio
import tempfile
import hashlib
import base64
import json
import csv
import mimetypes
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from fastembed import TextEmbedding

from langchain_community.document_loaders import PyPDFLoader

# Optional document-format libraries
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from openpyxl import load_workbook
    XLSX_AVAILABLE = True
except ImportError:
    XLSX_AVAILABLE = False

try:
    from pptx import Presentation
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai import AsyncOpenAI, OpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled

import edge_tts

# Local speech-to-text
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv(
    "GROQ_BASE_URL",
    "https://api.groq.com/openai/v1"
)

# IMPORTANT:
# Put the Groq model that already works for your account in .env.
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)
GROQ_VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL",
    "qwen/qwen3.6-27b"
)
VISUAL_DPI = int(os.getenv("VISUAL_DPI", "110"))
MAX_VISUAL_PAGES_PER_PDF = int(os.getenv("MAX_VISUAL_PAGES_PER_PDF", "30"))
VISUAL_CACHE_DIR = Path(os.getenv("VISUAL_CACHE_DIR", ".visual_cache"))
VISUAL_CACHE_DIR.mkdir(exist_ok=True)

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

COLLECTION_NAME = "voice-rag-agent"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384

# Local Whisper model.
# "tiny" is lighter/faster. "base" is generally better but uses more RAM.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")

VOICE_CONFIG = {
    "English": {
        "voice": "en-US-AriaNeural",
        "language": "en"
    },
    "Telugu": {
        "voice": "te-IN-ShrutiNeural",
        "language": "te"
    },
    "Tamil": {
        "voice": "ta-IN-PallaviNeural",
        "language": "ta"
    },
    "Hindi": {
        "voice": "hi-IN-SwaraNeural",
        "language": "hi"
    }
}


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Multilingual Conversational Voice RAG",
    page_icon="🎙️",
    layout="wide"
)

# ============================================================
# ============================================================
# AESTHETIC FRONTEND THEME — BLACK + VIOLET / 3D GLASS
# ============================================================
# Presentation-only styling. RAG, retrieval, agents, voice and
# Qdrant logic remain unchanged.
st.markdown(r"""
<style>
:root {
    --bg: #05030a;
    --panel: rgba(14, 8, 25, 0.78);
    --panel-strong: rgba(18, 10, 32, 0.92);
    --violet: #8b5cf6;
    --violet-2: #a855f7;
    --violet-3: #c084fc;
    --text: #f5f3ff;
    --muted: #aaa1bd;
}

/* ============================================================
   MAIN BACKGROUND
   ============================================================ */
.stApp {
    min-height: 100vh;
    color: var(--text);
    background:
        radial-gradient(circle at 50% 45%, rgba(139,92,246,.12), transparent 30%),
        radial-gradient(circle at 10% 10%, rgba(168,85,247,.10), transparent 25%),
        radial-gradient(circle at 90% 15%, rgba(124,58,237,.10), transparent 25%),
        radial-gradient(circle at 80% 90%, rgba(192,132,252,.08), transparent 28%),
        linear-gradient(135deg, #020106 0%, #07030e 42%, #030107 100%);
    background-attachment: fixed;
    overflow-x: hidden;
}

/* Animated ambient violet lights */
.stApp::before,
.stApp::after {
    content: "";
    position: fixed;
    width: 520px;
    height: 520px;
    border-radius: 50%;
    pointer-events: none;
    filter: blur(95px);
    opacity: .18;
    z-index: 0;
}

.stApp::before {
    background: #7c3aed;
    top: -220px;
    left: -180px;
    animation: violetOrbOne 15s ease-in-out infinite alternate;
}

.stApp::after {
    background: #c026d3;
    right: -220px;
    bottom: -220px;
    animation: violetOrbTwo 18s ease-in-out infinite alternate;
}

@keyframes violetOrbOne {
    0%   { transform: translate(0,0) scale(.9); }
    50%  { transform: translate(220px,150px) scale(1.2); }
    100% { transform: translate(80px,330px) scale(.95); }
}

@keyframes violetOrbTwo {
    0%   { transform: translate(0,0) scale(1); }
    50%  { transform: translate(-210px,-150px) scale(1.25); }
    100% { transform: translate(-40px,-280px) scale(.9); }
}

/* ============================================================
   3D GRID / CYBER FLOOR
   ============================================================ */
.stApp .main::after {
    content: "";
    position: fixed;
    left: -10%;
    right: -10%;
    bottom: -145px;
    height: 330px;
    pointer-events: none;
    opacity: .16;
    background-image:
        linear-gradient(rgba(139,92,246,.38) 1px, transparent 1px),
        linear-gradient(90deg, rgba(168,85,247,.32) 1px, transparent 1px);
    background-size: 46px 46px;
    transform:
        perspective(520px)
        rotateX(61deg)
        translateY(70px);
    transform-origin: bottom;
    z-index: 0;
    mask-image: linear-gradient(to top, black, transparent);
}

/* Soft cursor-like central glow. It slowly travels with the
   animated background even when JS is unavailable. */
.stApp .main::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background:
        radial-gradient(circle at 30% 30%, rgba(168,85,247,.07), transparent 22%),
        radial-gradient(circle at 70% 65%, rgba(139,92,246,.06), transparent 25%);
    animation: ambientShift 10s ease-in-out infinite alternate;
}

@keyframes ambientShift {
    0%   { transform: translate3d(-3%, -2%, 0) scale(1); }
    100% { transform: translate3d(3%, 2%, 0) scale(1.08); }
}

/* ============================================================
   CONTENT LAYER
   ============================================================ */
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    position: relative;
    z-index: 2;
}

/* ============================================================
   SIDEBAR — NO API KEY / URL FIELDS
   ============================================================ */
[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg,
            rgba(7,4,13,.97),
            rgba(12,6,22,.96));
    border-right: 1px solid rgba(139,92,246,.18);
    box-shadow: 12px 0 45px rgba(0,0,0,.45);
}

[data-testid="stSidebar"] * {
    color: #eee8ff;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(139,92,246,.16);
}

/* ============================================================
   TITLES
   ============================================================ */
h1 {
    background: linear-gradient(
        90deg,
        #ffffff 0%,
        #c084fc 25%,
        #8b5cf6 50%,
        #e9d5ff 75%,
        #ffffff 100%
    );
    background-size: 300% auto;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
    animation: titleFlow 7s linear infinite;
    font-weight: 850 !important;
    letter-spacing: -1.7px;
    text-shadow: 0 0 35px rgba(139,92,246,.18);
}

h2, h3 {
    color: #f5f3ff !important;
}

@keyframes titleFlow {
    from { background-position: 0% center; }
    to   { background-position: 300% center; }
}

/* ============================================================
   GLASS / 3D CARDS
   ============================================================ */
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stExpander"],
[data-testid="stForm"] {
    background:
        linear-gradient(145deg,
            rgba(24,13,39,.82),
            rgba(9,5,16,.72));
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(167,139,250,.16);
    border-radius: 20px;
    box-shadow:
        0 18px 55px rgba(0,0,0,.30),
        inset 0 1px 0 rgba(255,255,255,.06),
        0 0 0 1px rgba(139,92,246,.03);
    transition:
        transform .28s ease,
        box-shadow .28s ease,
        border-color .28s ease;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover,
[data-testid="stForm"]:hover {
    transform: translateY(-5px);
    border-color: rgba(168,85,247,.34);
    box-shadow:
        0 25px 65px rgba(0,0,0,.42),
        0 0 35px rgba(139,92,246,.10),
        inset 0 1px 0 rgba(255,255,255,.08);
}

/* Metric cards */
[data-testid="stMetric"] {
    background:
        linear-gradient(145deg,
            rgba(26,14,44,.90),
            rgba(10,6,18,.78));
    border: 1px solid rgba(139,92,246,.20);
    border-radius: 17px;
    padding: 14px;
    box-shadow:
        0 12px 32px rgba(0,0,0,.28),
        inset 0 1px 0 rgba(255,255,255,.05);
    transition: all .25s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-6px) rotateX(3deg) rotateY(-1deg);
    border-color: rgba(192,132,252,.35);
    box-shadow:
        0 20px 42px rgba(0,0,0,.38),
        0 0 28px rgba(139,92,246,.12);
}

/* ============================================================
   BUTTONS
   ============================================================ */
.stButton > button {
    border: 1px solid rgba(192,132,252,.22);
    border-radius: 13px;
    background:
        linear-gradient(135deg, #6d28d9 0%, #8b5cf6 48%, #a855f7 100%);
    color: white !important;
    font-weight: 750;
    box-shadow:
        0 10px 28px rgba(76,29,149,.32),
        inset 0 1px 0 rgba(255,255,255,.12);
    transition: all .22s ease;
    position: relative;
    overflow: hidden;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.015);
    box-shadow:
        0 16px 38px rgba(76,29,149,.42),
        0 0 28px rgba(168,85,247,.18);
    border-color: rgba(216,180,254,.42);
}

/* ============================================================
   INPUTS / SELECTS
   ============================================================ */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background: rgba(8,5,14,.88) !important;
    color: #f5f3ff !important;
    border: 1px solid rgba(139,92,246,.20) !important;
    border-radius: 13px !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus {
    border-color: #a855f7 !important;
    box-shadow:
        0 0 0 3px rgba(168,85,247,.12),
        0 0 24px rgba(139,92,246,.12) !important;
}

[data-baseweb="select"] > div {
    background: rgba(8,5,14,.90) !important;
    border-color: rgba(139,92,246,.20) !important;
    border-radius: 13px !important;
    color: #f5f3ff !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background:
        linear-gradient(145deg,
            rgba(20,10,34,.72),
            rgba(7,4,13,.72));
    border: 2px dashed rgba(139,92,246,.28);
    border-radius: 19px;
    padding: 9px;
    transition: all .3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #a855f7;
    background: rgba(24,12,40,.90);
    transform: translateY(-2px);
    box-shadow: 0 15px 35px rgba(76,29,149,.18);
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #a99fba !important;
    font-weight: 650;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #c084fc !important;
}

/* Status / alerts */
[data-testid="stAlert"] {
    background: rgba(16,8,27,.82) !important;
    border: 1px solid rgba(139,92,246,.18) !important;
    border-radius: 15px !important;
}

/* Captions */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #a9a0b8 !important;
}

/* ============================================================
   TOP HERO
   ============================================================ */
.violet-hero {
    margin: 0 0 1.4rem 0;
    padding: 20px 24px;
    border-radius: 20px;
    border: 1px solid rgba(168,85,247,.20);
    background:
        radial-gradient(circle at 10% 0%, rgba(168,85,247,.18), transparent 35%),
        linear-gradient(110deg,
            rgba(23,10,39,.88),
            rgba(8,4,15,.72));
    box-shadow:
        0 18px 50px rgba(0,0,0,.30),
        inset 0 1px 0 rgba(255,255,255,.06);
    position: relative;
    overflow: hidden;
}

.violet-hero::after {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    right: -65px;
    top: -85px;
    border-radius: 50%;
    background: rgba(168,85,247,.18);
    filter: blur(35px);
    animation: heroGlow 6s ease-in-out infinite alternate;
}

@keyframes heroGlow {
    from { transform: translate(0,0) scale(.9); }
    to { transform: translate(-70px,90px) scale(1.25); }
}

.violet-hero .eyebrow {
    font-size: 13px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #c084fc;
    font-weight: 750;
}

.violet-hero .subtitle {
    margin-top: 7px;
    color: #b8aec7;
    font-size: 14px;
}

/* Reduced motion accessibility */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
    }
}

@media (max-width: 900px) {
    .block-container {
        padding-top: 1rem;
    }
    h1 {
        font-size: 2rem !important;
    }
}

/* ===== V2 RESPONSIVE TOP / HERO ===== */
.hero-section {
    min-height: clamp(180px, 28vh, 300px) !important;
    height: auto !important;
    padding: clamp(22px, 4vh, 48px) clamp(18px, 4vw, 56px) !important;
    margin-top: clamp(4px, 1vh, 14px) !important;
    margin-bottom: clamp(12px, 2vh, 24px) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.hero-section h1 {
    font-size: clamp(2rem, 4.2vw, 4.5rem) !important;
    line-height: 1.02 !important;
    margin: 0 !important;
}

.hero-section p {
    font-size: clamp(.85rem, 1.2vw, 1.1rem) !important;
    margin-top: clamp(8px, 1.2vh, 16px) !important;
}

@media (max-height: 800px) {
    .hero-section {
        min-height: 150px !important;
        padding: 18px 28px !important;
        margin-bottom: 12px !important;
    }
    .hero-section h1 {
        font-size: clamp(1.8rem, 4vw, 3.2rem) !important;
    }
}

@media (max-width: 900px) {
    .hero-section {
        min-height: 160px !important;
        padding: 20px !important;
    }
}

/* ===== POPPING UPLOAD EXPERIENCE ===== */
div[data-testid="stFileUploader"] {
    position: relative !important;
    overflow: visible !important;
    border-radius: 22px !important;
    transition: transform .28s cubic-bezier(.2,.8,.2,1),
                filter .28s ease,
                box-shadow .28s ease !important;
}

div[data-testid="stFileUploader"]:hover {
    transform: translateY(-5px) scale(1.012) !important;
    filter: brightness(1.08) !important;
    box-shadow:
        0 0 0 1px rgba(168,85,247,.35),
        0 15px 45px rgba(124,58,237,.24),
        0 0 65px rgba(168,85,247,.12) !important;
}

/* Animated violet rings behind the upload box */
div[data-testid="stFileUploader"]::before,
div[data-testid="stFileUploader"]::after {
    content: "" !important;
    position: absolute !important;
    inset: -7px !important;
    border-radius: 25px !important;
    pointer-events: none !important;
    z-index: -1 !important;
}

div[data-testid="stFileUploader"]::before {
    border: 1px solid rgba(168,85,247,.28) !important;
    animation: uploadPulse 2.4s ease-out infinite !important;
}

div[data-testid="stFileUploader"]::after {
    border: 1px solid rgba(139,92,246,.18) !important;
    animation: uploadPulse 2.4s ease-out 1.2s infinite !important;
}

@keyframes uploadPulse {
    0% { transform: scale(.97); opacity: .75; }
    70% { transform: scale(1.045); opacity: 0; }
    100% { transform: scale(1.045); opacity: 0; }
}

/* File uploader drop-zone */
div[data-testid="stFileUploader"] section {
    border: 1px dashed rgba(168,85,247,.48) !important;
    border-radius: 20px !important;
    background:
        radial-gradient(circle at 50% 35%, rgba(139,92,246,.12), transparent 45%),
        rgba(10, 7, 20, .72) !important;
    backdrop-filter: blur(14px) !important;
    transition: all .3s ease !important;
}

div[data-testid="stFileUploader"] section:hover {
    border-color: rgba(192,132,252,.9) !important;
    background:
        radial-gradient(circle at 50% 35%, rgba(168,85,247,.20), transparent 52%),
        rgba(13, 8, 25, .82) !important;
}

/* Make the uploader browse button feel like a glowing capsule */
div[data-testid="stFileUploader"] button {
    border-radius: 999px !important;
    border: 1px solid rgba(192,132,252,.55) !important;
    background: linear-gradient(135deg, rgba(124,58,237,.95), rgba(168,85,247,.9)) !important;
    box-shadow: 0 6px 25px rgba(124,58,237,.28) !important;
    transition: transform .22s ease, box-shadow .22s ease !important;
}

div[data-testid="stFileUploader"] button:hover {
    transform: translateY(-2px) scale(1.035) !important;
    box-shadow:
        0 9px 30px rgba(124,58,237,.45),
        0 0 25px rgba(168,85,247,.25) !important;
}

/* Upload icon gently floats */
div[data-testid="stFileUploader"] svg {
    animation: uploadFloat 2.2s ease-in-out infinite !important;
}

@keyframes uploadFloat {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    50% { transform: translateY(-4px) rotate(-2deg); }
}

/* Short click/burst animation */
@keyframes uploadPop {
    0% { transform: scale(1); }
    35% { transform: scale(1.055); }
    65% { transform: scale(.985); }
    100% { transform: scale(1); }
}

.upload-pop {
    animation: uploadPop .42s cubic-bezier(.2,.9,.25,1) !important;
}

@media (prefers-reduced-motion: reduce) {
    div[data-testid="stFileUploader"]::before,
    div[data-testid="stFileUploader"]::after,
    div[data-testid="stFileUploader"] svg {
        animation: none !important;
    }
}


/* ============================================================
   V3 — STREAMLIT TOP BAR SAFE AREA
   Prevents the fixed Streamlit toolbar from covering the hero.
   ============================================================ */

/* Keep the native Streamlit top bar, but make it blend into the UI. */
[data-testid="stHeader"] {
    background: rgba(5, 3, 10, .82) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border-bottom: 1px solid rgba(139,92,246,.10) !important;
    z-index: 1000 !important;
}

/* The main content must start below the fixed toolbar. */
.block-container {
    padding-top: clamp(4.8rem, 7vh, 6.5rem) !important;
}

/* Give the hero a clean breathing space below the toolbar. */
.violet-hero {
    margin-top: 0 !important;
    padding-top: clamp(22px, 3vh, 34px) !important;
}

/* Don't let the hero's glow visually cross over the toolbar. */
.violet-hero {
    isolation: isolate;
}

/* Keep the main title comfortably below the top bar. */
h1 {
    margin-top: clamp(.35rem, 1vh, .8rem) !important;
}

/* Laptop-height optimization */
@media (max-height: 800px) {
    .block-container {
        padding-top: 4.4rem !important;
    }

    .violet-hero {
        padding-top: 18px !important;
        padding-bottom: 18px !important;
        margin-bottom: 1rem !important;
    }

    h1 {
        margin-bottom: .25rem !important;
    }
}

/* Smaller screens */
@media (max-width: 900px) {
    .block-container {
        padding-top: 4.2rem !important;
    }
}

/* Very small laptop/browser heights */
@media (max-height: 680px) {
    .block-container {
        padding-top: 3.9rem !important;
    }

    .violet-hero {
        padding-top: 14px !important;
        padding-bottom: 14px !important;
    }
}

</style>

<div class="violet-hero">
    <div class="eyebrow">Multimodal • Multilingual • Agentic RAG</div>
    <div class="subtitle">
        Intelligent document understanding with text, voice and visual evidence.
    </div>
</div>
""", unsafe_allow_html=True)

st.title("🎙️ Multilingual Conversational Voice RAG Agent")
st.caption(
    "Multi-PDF RAG • Qdrant • Groq Agents • Memory • Voice • Multilingual Output • Evaluation"
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "processed_files": {},
    "collection_ready": False,
    "chat_history": [],
    "last_retrieval": [],
    "last_answer": "",
    "last_speech": "",
    "last_language": "English",
    "last_question": "",
    "last_evaluation": {},
    "last_audio_path": None,
    "last_visual_evidence": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    # API credentials are intentionally NOT displayed in the UI.
    # They are loaded securely from .env at startup.
    st.success("🔐 API configuration loaded from .env")

    st.caption(
        "Qdrant and Groq credentials are hidden from the interface "
        "and are not rendered in the sidebar."
    )

    qdrant_url = QDRANT_URL.strip()
    qdrant_api_key = QDRANT_API_KEY.strip()
    groq_api_key = GROQ_API_KEY.strip()
    groq_model = GROQ_MODEL.strip()

    st.divider()

    selected_language = st.selectbox(
        "🌐 Response Language",
        ["English", "Telugu", "Tamil", "Hindi"]
    )

    operation_mode = st.selectbox(
        "🧠 AI Task",
        [
            "Question Answering",
            "Document Summary",
            "Key Points",
            "Important Dates",
            "Risks and Policies",
            "Explain Diagram",
            "Diagram to Structured Text"
        ]
    )

    retrieval_count = st.slider(
        "Retrieved chunks",
        min_value=3,
        max_value=10,
        value=5
    )

    st.divider()

    st.subheader("🎤 Voice Input")

    if WHISPER_AVAILABLE:
        st.success("Local Whisper STT is available.")
    else:
        st.warning(
            "Install faster-whisper to enable microphone input."
        )

    whisper_size = st.selectbox(
        "Whisper model",
        ["tiny", "base"],
        index=0
    )

    st.divider()
    st.subheader("🖼️ Visual / Multimodal RAG")
    enable_visual = st.checkbox(
        "Enable image and diagram understanding",
        value=True
    )
    vision_model = GROQ_VISION_MODEL.strip()
    st.caption(
        "Vision configuration is loaded from .env and hidden from the UI."
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):
        st.session_state.chat_history = []
        st.rerun()

    if st.button(
        "🧹 Clear Knowledge Base",
        use_container_width=True
    ):
        if qdrant_url and qdrant_api_key:
            try:
                client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key
                )
                client.delete_collection(
                    collection_name=COLLECTION_NAME
                )
                st.session_state.processed_files = {}
                st.session_state.collection_ready = False
                st.session_state.last_retrieval = []
                st.success("Knowledge base cleared.")
            except Exception as e:
                # Collection may not exist.
                st.warning(f"Could not clear collection: {e}")


# ============================================================
# CACHED RESOURCES
# ============================================================

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return TextEmbedding(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_whisper_model(model_size):
    if not WHISPER_AVAILABLE:
        return None

    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8"
    )


# ============================================================
# QDRANT HELPERS
# ============================================================

def get_client(url, api_key):
    return QdrantClient(
        url=url,
        api_key=api_key
    )


def ensure_collection(client):
    collections = client.get_collections().collections
    names = [item.name for item in collections]

    if COLLECTION_NAME not in names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

    return True


# ============================================================
# MULTI-FORMAT DOCUMENT INGESTION + VISUAL EXTRACTION
# ============================================================

SUPPORTED_EXTENSIONS = [
    "pdf", "docx", "txt", "md", "csv", "json", "xlsx", "pptx",
    "png", "jpg", "jpeg", "webp"
]
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def _stable_document_id(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def _image_data_url(image_bytes, extension):
    mime = mimetypes.guess_type(f"file.{extension}")[0] or "image/png"
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def analyze_image_with_vision(image_bytes, extension, api_key, model_name, prompt=None):
    """Describe an image/diagram using a vision-capable Groq model."""
    if not api_key:
        raise ValueError("Groq API key is required for visual understanding.")

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    image_url = _image_data_url(image_bytes, extension)
    instruction = prompt or (
        "Analyze this image as evidence for a document RAG system. "
        "Describe diagrams, flowcharts, architecture, labels, arrows, tables, "
        "relationships, numbers and visible text. Do not invent anything. "
        "Return detailed structured plain text that can be embedded and retrieved."
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }],
        temperature=0.1,
        max_tokens=1800
    )
    return response.choices[0].message.content.strip()


def _extract_text_from_file(file_bytes, file_name):
    """Return [(text, page_or_section, metadata)] for common office/data formats."""
    ext = Path(file_name).suffix.lower().lstrip(".")
    results = []

    if ext == "pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(file_bytes)
            path = f.name
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            for doc in docs:
                text = doc.page_content.strip()
                if text:
                    results.append((text, int(doc.metadata.get("page", 0)), {"format": "pdf"}))
        finally:
            try: os.remove(path)
            except OSError: pass

    elif ext == "docx":
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx is required for DOCX files.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f:
            f.write(file_bytes); path = f.name
        try:
            doc = DocxDocument(path)
            parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                rows = []
                for row in table.rows:
                    rows.append(" | ".join(cell.text.strip() for cell in row.cells))
                if rows: parts.append("TABLE:\n" + "\n".join(rows))
            text = "\n".join(parts).strip()
            if text: results.append((text, 0, {"format": "docx"}))
        finally:
            try: os.remove(path)
            except OSError: pass

    elif ext in {"txt", "md"}:
        text = file_bytes.decode("utf-8", errors="ignore").strip()
        if text: results.append((text, 0, {"format": ext}))

    elif ext == "json":
        raw = file_bytes.decode("utf-8", errors="ignore")
        try:
            obj = json.loads(raw)
            text = json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            text = raw
        if text.strip(): results.append((text.strip(), 0, {"format": "json"}))

    elif ext == "csv":
        raw = file_bytes.decode("utf-8-sig", errors="ignore")
        rows = list(csv.reader(raw.splitlines()))
        text = "\n".join(" | ".join(row) for row in rows)
        if text.strip(): results.append((text.strip(), 0, {"format": "csv"}))

    elif ext == "xlsx":
        if not XLSX_AVAILABLE:
            raise ImportError("openpyxl is required for XLSX files.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as f:
            f.write(file_bytes); path = f.name
        try:
            wb = load_workbook(path, read_only=True, data_only=True)
            for ws in wb.worksheets:
                rows = []
                for row in ws.iter_rows(values_only=True):
                    vals = ["" if v is None else str(v) for v in row]
                    if any(vals): rows.append(" | ".join(vals))
                if rows:
                    results.append((f"SHEET: {ws.title}\n" + "\n".join(rows), 0, {"format": "xlsx", "sheet": ws.title}))
            wb.close()
        finally:
            try: os.remove(path)
            except OSError: pass

    elif ext == "pptx":
        if not PPTX_AVAILABLE:
            raise ImportError("python-pptx is required for PPTX files.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as f:
            f.write(file_bytes); path = f.name
        try:
            prs = Presentation(path)
            for idx, slide in enumerate(prs.slides):
                texts = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        texts.append(shape.text.strip())
                    if getattr(shape, "has_table", False):
                        for row in shape.table.rows:
                            texts.append(" | ".join(cell.text.strip() for cell in row.cells))
                if texts:
                    results.append(("SLIDE " + str(idx + 1) + "\n" + "\n".join(texts), idx, {"format": "pptx", "slide": idx + 1}))
        finally:
            try: os.remove(path)
            except OSError: pass

    return results


def _render_pdf_visual_pages(file_bytes, document_id):
    """Render PDF pages that contain images/drawings for visual RAG."""
    if not PYMUPDF_AVAILABLE:
        return []
    pdf_path = VISUAL_CACHE_DIR / f"{document_id}.pdf"
    pdf_path.write_bytes(file_bytes)
    results = []
    try:
        doc = fitz.open(pdf_path)
        for page_no in range(min(len(doc), MAX_VISUAL_PAGES_PER_PDF)):
            page = doc[page_no]
            has_images = len(page.get_images(full=True)) > 0
            has_drawings = len(page.get_drawings()) >= 8
            if not (has_images or has_drawings):
                continue
            pix = page.get_pixmap(dpi=VISUAL_DPI, alpha=False)
            image_path = VISUAL_CACHE_DIR / f"{document_id}_page_{page_no + 1}.png"
            pix.save(image_path)
            results.append((page_no, image_path))
        doc.close()
    except Exception:
        pass
    return results


def process_file(uploaded_file, client, embedder, vision_enabled, vision_model, groq_api_key):
    """Ingest PDF, DOCX, TXT, MD, CSV, JSON, XLSX, PPTX and images."""
    ensure_collection(client)
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    ext = Path(file_name).suffix.lower().lstrip(".")
    document_id = _stable_document_id(file_bytes)

    if document_id in st.session_state.processed_files:
        return 0, 0, "already_processed"

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: .{ext}")

    records = []
    visual_records = []

    # Direct image upload -> vision description.
    if ext in IMAGE_EXTENSIONS:
        if not vision_enabled:
            raise ValueError("Enable visual understanding to ingest image files.")
        description = analyze_image_with_vision(file_bytes, ext, groq_api_key, vision_model)
        records.append((description, 0, {"format": ext, "content_type": "visual", "visual_type": "image"}))
        image_path = VISUAL_CACHE_DIR / f"{document_id}.{ext}"
        image_path.write_bytes(file_bytes)
        visual_records.append((description, 0, image_path, "image"))
    else:
        records.extend(_extract_text_from_file(file_bytes, file_name))

        # PDF visual pages are separately analyzed and stored as visual evidence.
        if ext == "pdf" and vision_enabled and groq_api_key and PYMUPDF_AVAILABLE:
            for page_no, image_path in _render_pdf_visual_pages(file_bytes, document_id):
                try:
                    desc = analyze_image_with_vision(
                        image_path.read_bytes(), "png", groq_api_key, vision_model,
                        prompt=("Analyze this PDF page as a visual document page. "
                                "Identify diagrams, flowcharts, architecture, tables, labels, arrows, "
                                "relationships and important visible text. Focus on information not reliably "
                                "captured by normal PDF text extraction. Do not invent details.")
                    )
                    visual_records.append((desc, page_no, image_path, "pdf_page_visual"))
                except Exception as e:
                    st.warning(f"Visual analysis skipped for page {page_no + 1}: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    text_docs = []
    for text, page, meta in records:
        # lightweight object compatible with the existing chunking approach
        from langchain_core.documents import Document
        text_docs.append(Document(page_content=text, metadata={"page": page, **meta}))

    chunks = splitter.split_documents(text_docs)
    valid = [(chunk, chunk.page_content.strip()) for chunk in chunks if chunk.page_content.strip()]

    texts = [text for _, text in valid] + [item[0] for item in visual_records]
    if not texts:
        raise ValueError("No readable content was found in this file.")

    vectors = list(embedder.embed(texts))
    points = []
    vector_index = 0

    for index, (chunk, text) in enumerate(valid):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}-text-{index}"))
        payload = {
            "text": text,
            "source": file_name,
            "document_id": document_id,
            "page": int(chunk.metadata.get("page", 0)),
            "chunk_index": index,
            "content_type": chunk.metadata.get("content_type", "text"),
            "format": ext,
            "sheet": chunk.metadata.get("sheet", ""),
            "slide": chunk.metadata.get("slide", "")
        }
        points.append(PointStruct(id=point_id, vector=vectors[vector_index].tolist(), payload=payload))
        vector_index += 1

    for visual_index, (desc, page_no, image_path, visual_type) in enumerate(visual_records):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}-visual-{visual_index}"))
        payload = {
            "text": desc,
            "source": file_name,
            "document_id": document_id,
            "page": int(page_no),
            "chunk_index": visual_index,
            "content_type": "visual",
            "visual_type": visual_type,
            "visual_image_path": str(image_path),
            "format": ext
        }
        points.append(PointStruct(id=point_id, vector=vectors[vector_index].tolist(), payload=payload))
        vector_index += 1

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    st.session_state.processed_files[document_id] = file_name
    return len(valid), len(visual_records), "processed"


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_context(query, client, embedder, limit=5, include_visual=True):
    query_vector = list(embedder.embed([query]))[0]
    candidate_limit = max(limit * 5, 25)
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=candidate_limit,
        with_payload=True
    )
    points = list(getattr(result, "points", []))
    if not points:
        return []

    # Hybrid reranking: semantic similarity + token overlap + visual preference for diagram tasks.
    query_tokens = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", query.lower()))
    diagram_terms = {"diagram", "flowchart", "architecture", "chart", "figure", "image", "visual", "draw", "structure"}
    visual_query = bool(query_tokens & diagram_terms)

    ranked = []
    for p in points:
        payload = p.payload or {}
        text = str(payload.get("text", ""))
        text_tokens = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower()))
        lexical = len(query_tokens & text_tokens) / max(1, len(query_tokens))
        semantic = float(p.score)
        final_score = 0.82 * semantic + 0.18 * lexical
        if visual_query and payload.get("content_type") == "visual":
            final_score += 0.08
        if not include_visual and payload.get("content_type") == "visual":
            continue
        ranked.append((final_score, semantic, p))

    ranked.sort(key=lambda x: x[0], reverse=True)
    selected = []
    per_doc = {}
    max_per_doc = 2 if limit >= 5 else 1
    for _, _, p in ranked:
        doc_id = (p.payload or {}).get("document_id", "")
        if per_doc.get(doc_id, 0) >= max_per_doc:
            continue
        selected.append(p)
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        if len(selected) >= limit:
            break

    # Fill remaining slots if diversity constraint was too strict.
    if len(selected) < limit:
        used = {str(p.id) for p in selected}
        for _, _, p in ranked:
            if str(p.id) in used:
                continue
            selected.append(p)
            if len(selected) >= limit:
                break

    return selected


def build_context(matches):
    context_parts = []

    for rank, match in enumerate(matches, 1):
        payload = match.payload or {}

        source = payload.get(
            "source",
            "Unknown source"
        )

        page = int(
            payload.get("page", 0)
        )

        text = payload.get(
            "text",
            ""
        )
        content_type = payload.get("content_type", "text")
        visual_type = payload.get("visual_type", "")

        context_parts.append(
            f"""
[DOCUMENT CHUNK {rank}]
Source: {source}
Page: {page + 1}
Content type: {content_type} {visual_type}
Similarity score: {match.score:.4f}

{text}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# AGENTS
# ============================================================

def create_agents(api_key, model_name):
    groq_client = AsyncOpenAI(
        api_key=api_key,
        base_url=GROQ_BASE_URL
    )

    set_tracing_disabled(True)

    model = OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=groq_client
    )

    answer_agent = Agent(
        name="Documentation Answer Agent",
        model=model,
        instructions="""
You are a grounded document intelligence agent.

Use ONLY the supplied retrieved document context.

Rules:
1. Never invent facts.
2. If the context does not contain the answer, say:
   "I could not find that information in the uploaded documents."
3. Answer in the requested language.
4. Preserve names, dates and numbers accurately.
5. For comparisons, use only information present in context.
6. For calculations, show the calculation briefly.
7. When useful, mention the source document and page.
8. Do not claim that information exists if it is not in the context.
"""
    )

    speech_agent = Agent(
        name="Speech Preparation Agent",
        model=model,
        instructions="""
Prepare the answer for text-to-speech.

Rules:
1. Preserve the exact meaning.
2. Keep the requested language.
3. Remove markdown, bullets and unnecessary symbols.
4. Keep names, dates and numbers.
5. Do not add information.
6. Return only speech-ready text.
"""
    )

    return answer_agent, speech_agent


def make_prompt(
    question,
    context,
    language,
    operation_mode,
    chat_history
):
    history_text = ""

    if chat_history:
        recent = chat_history[-6:]
        history_text = "\n".join(
            [
                f"User: {item['user']}\n"
                f"Assistant: {item['assistant']}"
                for item in recent
            ]
        )

    if operation_mode == "Document Summary":
        task = """
Summarize the retrieved documents. Give the most important
information and do not invent anything.
"""

    elif operation_mode == "Key Points":
        task = """
Extract the most important key points from the retrieved
documents. Keep them concise and factual.
"""

    elif operation_mode == "Important Dates":
        task = """
Extract important dates, years, deadlines, launches or events
from the retrieved documents. Include the event and date.
"""

    elif operation_mode == "Explain Diagram":
        task = """Explain the retrieved diagram or visual evidence step by step. Describe components, labels, arrows, relationships, flow and purpose. Clearly separate visible evidence from inference."""

    elif operation_mode == "Diagram to Structured Text":
        task = """Convert the retrieved diagram or visual evidence into structured text using headings, components, connections, inputs, outputs and sequence. Do not invent missing labels."""

    elif operation_mode == "Risks and Policies":
        task = """
Identify risks, security rules, privacy statements, support
policies, restrictions or compliance-related information found
in the retrieved documents.
"""

    else:
        task = """
Answer the user's question directly using only the retrieved
document context.
"""

    return f"""
Requested response language: {language}

Task:
{task}

User question/request:
{question}

Recent conversation:
{history_text if history_text else "No previous conversation."}

Retrieved document context:
{context}

Return the answer in {language}.
"""


async def run_agents(
    answer_agent,
    speech_agent,
    prompt,
    language
):
    answer_result = await Runner.run(
        answer_agent,
        prompt
    )

    answer = answer_result.final_output.strip()

    speech_result = await Runner.run(
        speech_agent,
        f"""
Requested language: {language}

Answer:
{answer}

Prepare this answer for speech.
Do not translate it into another language.
Do not add information.
Return only speech-ready text.
"""
    )

    speech_text = speech_result.final_output.strip()

    return answer, speech_text


# ============================================================
# TEXT TO SPEECH
# ============================================================

async def generate_tts(
    text,
    voice,
    output_path
):
    communicator = edge_tts.Communicate(
        text=text,
        voice=voice
    )

    await communicator.save(output_path)


def create_voice(text, language):
    voice = VOICE_CONFIG[
        language
    ]["voice"]

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    )

    path = temp.name
    temp.close()

    asyncio.run(
        generate_tts(
            text,
            voice,
            path
        )
    )

    return path


# ============================================================
# SPEECH TO TEXT
# ============================================================

def transcribe_audio(
    audio_bytes,
    model_size
):
    if not WHISPER_AVAILABLE:
        raise RuntimeError(
            "faster-whisper is not installed."
        )

    model = get_whisper_model(
        model_size
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp:
        temp.write(audio_bytes)
        audio_path = temp.name

    try:
        segments, info = model.transcribe(
            audio_path,
            beam_size=5
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        return text, info.language

    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass


# ============================================================
# EVALUATION
# ============================================================

def evaluate_retrieval(matches):
    if not matches:
        return {
            "retrieval_score": 0.0,
            "context_coverage": 0.0
        }

    scores = [
        float(match.score)
        for match in matches
    ]

    # Qdrant cosine similarity is approximately [-1, 1].
    normalized = [
        max(0.0, min(1.0, (score + 1) / 2))
        for score in scores
    ]

    avg_score = sum(normalized) / len(normalized)

    strong_chunks = sum(
        1
        for score in normalized
        if score >= 0.65
    )

    coverage = strong_chunks / len(normalized)

    return {
        "retrieval_score": avg_score,
        "context_coverage": coverage
    }


def evaluate_answer(
    question,
    answer,
    context
):
    """
    Lightweight local heuristic evaluation.
    This is not a replacement for a formal RAG benchmark.
    """

    if not answer.strip():
        return {
            "answer_grounding": 0.0,
            "answer_relevance": 0.0
        }

    # Compare meaningful words in the answer with the context.
    answer_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]{4,}\b",
            answer.lower()
        )
    )

    context_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]{4,}\b",
            context.lower()
        )
    )

    overlap = (
        len(answer_words & context_words)
        / max(1, len(answer_words))
    )

    # Simple question/answer lexical overlap.
    question_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]{4,}\b",
            question.lower()
        )
    )

    relevance = (
        len(question_words & context_words)
        / max(1, len(question_words))
    )

    return {
        "answer_grounding": min(1.0, overlap),
        "answer_relevance": min(1.0, relevance)
    }


def run_evaluation(
    question,
    answer,
    context,
    matches
):
    retrieval = evaluate_retrieval(
        matches
    )

    answer_metrics = evaluate_answer(
        question,
        answer,
        context
    )

    metrics = {
        **retrieval,
        **answer_metrics
    }

    metrics["overall_score"] = sum(
        metrics.values()
    ) / len(metrics)

    return metrics


# ============================================================
# 1. MULTI-PDF KNOWLEDGE BASE
# ============================================================

st.header("1️⃣ Build Your Knowledge Base")

if not qdrant_url or not qdrant_api_key:
    st.warning(
        "Enter your Qdrant URL and API key in the sidebar."
    )
else:
    try:
        qdrant_client = get_client(
            qdrant_url,
            qdrant_api_key
        )

        embedding_model = get_embedding_model()

        ensure_collection(
            qdrant_client
        )

        uploaded_files = st.file_uploader(
            "Upload one or more documents / images",
            type=SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
            help="Supported: PDF, DOCX, TXT, MD, CSV, JSON, XLSX, PPTX, PNG, JPG, JPEG, WEBP"
        )

        if uploaded_files:

            if st.button(
                "📚 Process Uploaded PDFs",
                use_container_width=True
            ):

                total_chunks = 0

                with st.status(
                    "Building knowledge base...",
                    expanded=True
                ) as status:

                    for uploaded_file in uploaded_files:

                        st.write(
                            f"🔄 Processing {uploaded_file.name}..."
                        )

                        count, visual_count, result = process_file(
                            uploaded_file,
                            qdrant_client,
                            embedding_model,
                            enable_visual,
                            vision_model,
                            groq_api_key
                        )

                        if result == "already_processed":
                            st.write(
                                f"ℹ️ {uploaded_file.name} "
                                "is already in the knowledge base."
                            )
                        else:
                            st.write(
                                f"✅ {uploaded_file.name}: "
                                f"{count} text chunks + {visual_count} visual records added."
                            )

                            total_chunks += count

                    st.session_state.collection_ready = True

                    status.update(
                        label="Knowledge base ready!",
                        state="complete"
                    )

                st.success(
                    f"Added {total_chunks} new chunks."
                )

        if st.session_state.processed_files:
            st.subheader("📂 Documents in Knowledge Base")

            for document_name in (
                st.session_state.processed_files.values()
            ):
                st.write(f"• {document_name}")

    except Exception as e:
        st.error(
            f"Qdrant initialization failed: {e}"
        )


# ============================================================
# 2. VOICE INPUT
# ============================================================

st.header("2️⃣ Ask by Voice or Text")

voice_question = None

if hasattr(st, "audio_input") and WHISPER_AVAILABLE:
    audio_input = st.audio_input(
        "🎤 Record your question"
    )

    if audio_input is not None:

        if st.button(
            "🗣️ Transcribe Voice",
            use_container_width=True
        ):
            try:
                with st.spinner(
                    "Transcribing locally with Whisper..."
                ):
                    voice_question, detected_language = (
                        transcribe_audio(
                            audio_input.getvalue(),
                            whisper_size
                        )
                    )

                st.success(
                    f"Detected speech language: "
                    f"{detected_language}"
                )

                st.session_state.voice_question = (
                    voice_question
                )

            except Exception as e:
                st.error(
                    f"Voice transcription failed: {e}"
                )

if "voice_question" not in st.session_state:
    st.session_state.voice_question = ""

question = st.text_area(
    "Your question",
    value=st.session_state.voice_question,
    placeholder=(
        "Example: Which products were launched "
        "between 2021 and 2025?"
    ),
    height=90
)


# ============================================================
# 3. PROCESS QUERY
# ============================================================

if st.button(
    "🚀 Ask AI Agent",
    type="primary",
    use_container_width=True
):

    if not question.strip():
        st.warning(
            "Please enter a question or record your voice."
        )

    elif not st.session_state.collection_ready:
        st.error(
            "Please upload and process at least one PDF first."
        )

    elif not groq_api_key:
        st.error(
            "Groq API key is missing."
        )

    else:

        try:

            # -----------------------------------------------
            # STEP 1
            # -----------------------------------------------

            st.write(
                "🔄 **Step 1: Creating query embedding "
                "and searching Qdrant...**"
            )

            matches = retrieve_context(
                question,
                qdrant_client,
                embedding_model,
                limit=retrieval_count,
                include_visual=enable_visual
            )

            st.write(
                f"Generated embedding size: {VECTOR_SIZE}"
            )

            st.write(
                f"Found {len(matches)} relevant chunks."
            )

            # -----------------------------------------------
            # STEP 2
            # -----------------------------------------------

            st.write(
                "🔄 **Step 2: Preparing retrieved context...**"
            )

            context = build_context(
                matches
            )

            unique_sources = set()

            for rank, match in enumerate(
                matches,
                1
            ):
                payload = match.payload or {}

                source = payload.get(
                    "source",
                    "Unknown"
                )

                page = int(
                    payload.get("page", 0)
                ) + 1

                unique_sources.add(
                    (source, page)
                )

                st.write(
                    f"Retrieved chunk {rank}: "
                    f"{source}, page {page}"
                )

            # -----------------------------------------------
            # STEP 3
            # -----------------------------------------------

            st.write(
                "🔄 **Step 3: Setting up Groq agents...**"
            )

            answer_agent, speech_agent = create_agents(
                groq_api_key,
                groq_model
            )

            st.write(
                "Initialized Documentation Answer Agent "
                "and Speech Preparation Agent."
            )

            # -----------------------------------------------
            # STEP 4
            # -----------------------------------------------

            st.write(
                "🔄 **Step 4: Generating grounded answer "
                "with Groq...**"
            )

            prompt = make_prompt(
                question,
                context,
                selected_language,
                operation_mode,
                st.session_state.chat_history
            )

            answer, speech_text = asyncio.run(
                run_agents(
                    answer_agent,
                    speech_agent,
                    prompt,
                    selected_language
                )
            )

            st.write(
                f"Generated answer length: "
                f"{len(answer)} characters"
            )

            # -----------------------------------------------
            # STEP 5
            # -----------------------------------------------

            st.write(
                "🔄 **Step 5: Preparing speech text...**"
            )

            st.write(
                f"Speech language: {selected_language}"
            )

            st.write(
                f"Speech text length: "
                f"{len(speech_text)} characters"
            )

            # -----------------------------------------------
            # STEP 6
            # -----------------------------------------------

            st.write(
                f"🔄 **Step 6: Generating "
                f"{selected_language} voice...**"
            )

            audio_path = create_voice(
                speech_text,
                selected_language
            )

            st.write(
                "✅ Multilingual voice generated."
            )

            # -----------------------------------------------
            # STEP 7
            # -----------------------------------------------

            st.write(
                "🔄 **Step 7: Evaluating retrieval "
                "and answer quality...**"
            )

            metrics = run_evaluation(
                question,
                answer,
                context,
                matches
            )

            st.write(
                "✅ Evaluation completed."
            )

            # -----------------------------------------------
            # SAVE MEMORY
            # -----------------------------------------------

            st.session_state.chat_history.append(
                {
                    "user": question,
                    "assistant": answer,
                    "language": selected_language,
                    "time": datetime.now().strftime(
                        "%H:%M:%S"
                    )
                }
            )

            st.session_state.last_retrieval = matches
            st.session_state.last_answer = answer
            st.session_state.last_speech = speech_text
            st.session_state.last_language = selected_language
            st.session_state.last_question = question
            st.session_state.last_evaluation = metrics

            # -----------------------------------------------
            # RESULTS
            # -----------------------------------------------

            st.divider()

            st.subheader("📝 Answer")
            st.write(answer)

            st.subheader(
                f"🔊 {selected_language} Voice Response"
            )

            with open(
                audio_path,
                "rb"
            ) as audio_file:
                audio_bytes = audio_file.read()

            st.audio(
                audio_bytes,
                format="audio/mp3"
            )

            # -----------------------------------------------
            # SOURCES
            # -----------------------------------------------

            st.subheader("📚 Sources")

            for source, page in sorted(
                unique_sources
            ):
                st.write(
                    f"• 📄 {source} — page {page}"
                )

            # -----------------------------------------------
            # RETRIEVED CHUNKS
            # -----------------------------------------------

            with st.expander(
                "🔎 View Retrieved Context"
            ):
                for rank, match in enumerate(
                    matches,
                    1
                ):
                    payload = match.payload or {}

                    st.markdown(
                        f"**Chunk {rank}** — "
                        f"{payload.get('source', 'Unknown')} "
                        f"(page {int(payload.get('page', 0)) + 1})"
                    )

                    st.caption(
                        f"Similarity score: "
                        f"{match.score:.4f}"
                    )

                    st.write(
                        payload.get(
                            "text",
                            ""
                        )
                    )

            # -----------------------------------------------
            # EVALUATION DASHBOARD
            # -----------------------------------------------

            st.subheader("📊 RAG Evaluation")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Retrieval Score",
                f"{metrics['retrieval_score'] * 100:.1f}%"
            )

            c2.metric(
                "Context Coverage",
                f"{metrics['context_coverage'] * 100:.1f}%"
            )

            c3.metric(
                "Answer Grounding",
                f"{metrics['answer_grounding'] * 100:.1f}%"
            )

            c4.metric(
                "Overall",
                f"{metrics['overall_score'] * 100:.1f}%"
            )

            st.info(
                "These are lightweight local heuristic metrics "
                "for development. For a formal research evaluation, "
                "use a labelled test set and metrics such as "
                "retrieval recall, context precision, faithfulness "
                "and answer correctness."
            )

            # -----------------------------------------------
            # CONVERSATION MEMORY
            # -----------------------------------------------

            with st.expander(
                "💬 Conversation Memory"
            ):
                for item in st.session_state.chat_history:
                    st.markdown(
                        f"**You:** {item['user']}"
                    )
                    st.markdown(
                        f"**AI ({item['language']}):** "
                        f"{item['assistant']}"
                    )
                    st.caption(
                        item["time"]
                    )

            try:
                os.remove(audio_path)
            except OSError:
                pass

        except Exception as e:
            st.error(
                f"❌ Query processing failed: {e}"
            )
            st.exception(e)


# ============================================================
# PROJECT ARCHITECTURE
# ============================================================

with st.expander(
    "🏗️ View Project Architecture"
):

    st.markdown(
        """
### Complete Architecture

```text
                 ┌─────────────────────┐
                 │   Multiple PDFs      │
                 └──────────┬──────────┘
                            ↓
                    PDF Text Extraction
                            ↓
                       Chunking
                            ↓
                       FastEmbed
                            ↓
                    Qdrant Vector DB
                            ↑
                            │
                 User Question / Voice
                            ↓
                    Local Whisper STT
                            ↓
                       Text Query
                            ↓
                    Semantic Retrieval
                            ↓
                 Top-K Relevant Chunks
                            ↓
                    Groq Answer Agent
                            ↓
                 Grounded Answer
                            ↓
              ┌─────────────┴─────────────┐
              ↓                           ↓
       Conversation Memory          Source Citations
              ↓
       Speech Preparation Agent
              ↓
       English / Telugu / Tamil / Hindi
              ↓
        Edge Neural Text-to-Speech
              ↓
              🔊 Voice

              Parallel:
       Retrieval + Answer Evaluation
              ↓
          📊 Dashboard
```

### Features implemented

1. **Multiple PDF knowledge base**
2. **Source/page citations**
3. **Hallucination protection**
4. **Conversation memory**
5. **Voice input with local Whisper**
6. **Document intelligence modes**
7. **RAG evaluation dashboard**
8. **English + Telugu + Tamil + Hindi voice output**
"""
    )


# ============================================================
# SETUP HELP
# ============================================================

with st.expander("🛠️ Setup / Troubleshooting"):

    st.markdown(
        f"""
### `.env`

```env
GROQ_API_KEY=your_groq_key
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=your_working_groq_model

QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key

WHISPER_MODEL_SIZE=tiny
GROQ_VISION_MODEL=qwen/qwen3.6-27b
VISUAL_DPI=110
MAX_VISUAL_PAGES_PER_PDF=30
```

### Install packages

```powershell
pip install -r requirements_visual.txt
```

### Run

```powershell
python -m streamlit run rag_voice.py
```

### Voice input

The microphone uses local `faster-whisper`.

Current selected Whisper model:
`{whisper_size}`

The first use downloads the Whisper model. After that,
it is cached locally.

### Supported formats

PDF, DOCX, TXT, Markdown, CSV, JSON, XLSX, PPTX, PNG, JPG, JPEG and WEBP.
PDF pages containing diagrams/images can also be rendered and analyzed by the vision model.

### Important

Groq is used for the LLM and optional vision analysis.
Qdrant is used for vector retrieval.
FastEmbed runs locally.
Whisper runs locally for speech-to-text.
Edge TTS generates multilingual speech.
"""
    )
