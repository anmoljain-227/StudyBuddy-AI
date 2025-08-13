# app.py
import os
import re
import io
import time
import uuid
import json
from typing import List, Dict, Tuple
import streamlit as st
from PIL import Image
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from chromadb import PersistentClient
from dotenv import load_dotenv
import google.generativeai as genai
from chromadb.config import Settings

# ------------------------
# Config / constants
# ------------------------
CHROMA_PATH = "chroma_db"          # persistent directory for Chroma
COLLECTION_NAME = "documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_CHUNK_SIZE = 1000            # characters per chunk
EMBED_CHUNK_OVERLAP = 200

# ------------------------
# Load environment variables
# ------------------------
load_dotenv()

# ------------------------
# Utilities: file -> text
# ------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            text_pages.append(text)
    return "\n".join(text_pages)

def extract_text_from_docx(file_bytes: bytes) -> str:
    tmp_path = f"/tmp/{uuid.uuid4()}.docx"
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)
    doc = docx.Document(tmp_path)
    paragraphs = [p.text for p in doc.paragraphs]
    try:
        os.remove(tmp_path)
    except:
        pass
    return "\n".join(paragraphs)

def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except:
        return str(file_bytes)

def extract_text(file) -> Tuple[str, str]:
    """
    file: UploadedFile from streamlit
    returns (filename, extracted_text)
    """
    raw = file.read()
    name = file.name
    name_l = name.lower()
    if name_l.endswith(".pdf"):
        text = extract_text_from_pdf(raw)
    elif name_l.endswith(".docx"):
        text = extract_text_from_docx(raw)
    elif name_l.endswith(".txt") or name_l.endswith(".md"):
        text = extract_text_from_txt(raw)
    else:
        # fallback: try decode
        text = extract_text_from_txt(raw)
    return name, text

# ------------------------
# Text chunking
# ------------------------
def chunk_text(text: str, chunk_size:int=EMBED_CHUNK_SIZE, overlap:int=EMBED_CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# ------------------------
# Embedding model loader
# ------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    return SentenceTransformer(model_name)

# ------------------------
# Chroma DB client / collection
# ------------------------
@st.cache_resource(show_spinner=False)
def get_chroma_client_and_collection(persist_directory: str = CHROMA_PATH):
    os.makedirs(persist_directory, exist_ok=True)
    client = PersistentClient(path=persist_directory)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        collection = client.create_collection(name=COLLECTION_NAME)
    return client, collection

# ------------------------
# Upsert docs -> chroma
# ------------------------
def add_documents_to_chroma(collection, model, docs: List[Dict]):
    """
    docs: list of dicts with keys: id, text, metadata
    """
    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    metadatas = [d.get("metadata", {}) for d in docs]
    # compute embeddings in batches
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # chroma expects list of embeddings
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings.tolist())

# ------------------------
# Query chroma
# ------------------------
def query_chroma(collection, model, query: str, top_k:int = 4):
    # handle empty query: use a generic broad query
    q = query if query and query.strip() else "summarize documents"
    q_emb = model.encode([q], convert_to_numpy=True)[0].tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents","metadatas","distances"])
    docs = []
    if res and len(res.get("documents", [])) > 0:
        retrieved = res["documents"][0]
        metadatas = res["metadatas"][0]
        distances = res.get("distances", [[]])[0]
        for d, m, dist in zip(retrieved, metadatas, distances):
            docs.append({"text": d, "metadata": m, "distance": dist})
    return docs

# ------------------------
# Gemini (GenAI) wrapper
# ------------------------
def init_genai_client(api_key: str):
    """
    Initialize Gemini client using google-generativeai SDK.
    """
    if not api_key:
        raise ValueError("GEMINI API key missing")
    genai.configure(api_key=api_key)
    return True

def ask_gemini_with_context(model_name: str, question: str, context_snippets: List[Dict], max_output_tokens:int=512, temperature:float=0.0):
    """
    Build a context-aware prompt and query Gemini using the SDK.
    Returns the raw text response.
    """
    # Prepare context
    ctx_parts = []
    for i, sn in enumerate(context_snippets):
        md = sn.get("metadata", {})
        src = md.get("source", f"doc_{i}")
        ctx_parts.append(f"---\nSource: {src}\nContent:\n{sn['text']}\n")
    context_text = "\n".join(ctx_parts)

    prompt = f"""
You are an assistant that answers user questions using only the provided context.
If the answer is not present in the context, say "I don't know based on the provided files."
Keep answers concise and highlight sources when helpful.

Context:
{context_text}

Question:
{question}

Answer:
"""
    # call model
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature
            )
        )
        # response may have .text
        text = getattr(response, "text", None)
        if text is None:
            # try candidates
            candidates = getattr(response, "candidates", None)
            if candidates and len(candidates) > 0:
                text = candidates[0].content
            else:
                text = str(response)
        return text
    except Exception as e:
        # bubble up with informative message
        raise Exception(f"GenAI call error: {e}")

# ------------------------
# Helper for robust JSON parsing from model output
# ------------------------
# def extract_json_from_text(text: str):
#     """
#     Try to extract JSON array/object from the model text.
#     Returns Python object or raises.
#     """
#     # Remove triple backticks and language hints
#     text = text.strip()
#     if text.startswith("```"):
#         # strip the first code fence
#         text = text.split("```", 1)[1]
#         # remove leading language tags like 'json'
#         text = text.lstrip().lstrip("json").lstrip()

#     # quick attempt: direct json.loads
#     try:
#         return json.loads(text)
#     except Exception:
#         pass

#     # try to locate first { or [
#     start_idx = None
#     for ch in ('[', '{'):
#         idx = text.find(ch)
#         if idx != -1:
#             start_idx = idx
#             break
#     if start_idx is None:
#         raise ValueError("No JSON start found")

#     substr = text[start_idx:]
#     # Remove trailing code fences if present
#     if "```" in substr:
#         substr = substr.split("```")[0].strip()

#     # Attempt to load progressively shrinking suffixes until success
#     for end_pos in range(len(substr), 0, -1):
#         candidate = substr[:end_pos]
#         try:
#             return json.loads(candidate)
#         except Exception:
#             continue
#     raise ValueError("Failed to parse JSON from model output")



def clean_json_text(raw: str) -> str:
    """
    Cleans raw model output to extract only the JSON array or object.
    Removes markdown code fences and language hints.
    """
    # Remove triple backticks and any language specifier (like ```json)
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", raw.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip())

    # Extract the first [...] or {...} block
    match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    return cleaned.strip()
# ------------------------
# MCQ generation helper (asks Gemini to return JSON)
# ------------------------
def generate_mcq_json(model_name: str, topic: str, context_snippets: List[Dict], gemini_key: str, max_output_tokens=1024) -> List[Dict]:
    init_genai_client(gemini_key)

    if topic and topic.strip():
        q_text = (
            f"Generate 10 multiple-choice questions (4 options each) about the following topic: '{topic}'. "
            "Use only the provided context. Return ONLY a valid JSON array of 10 objects with fields: "
            "question (string), options (array of 4 strings), answer_index (0-3), explanation (string). "
            "Do NOT include code fences, markdown, or any explanatory text."
        )
    else:
        q_text = (
            "Generate 10 multiple-choice questions (4 options each) from the provided documents. "
            "Use only the provided context. Return ONLY a valid JSON array of 10 objects with fields: "
            "question (string), options (array of 4 strings), answer_index (0-3), explanation (string). "
            "Do NOT include code fences, markdown, or any explanatory text."
        )

    # Ask model
    raw = ask_gemini_with_context(
        model_name, q_text, context_snippets,
        max_output_tokens=max_output_tokens, temperature=0.0
    )

    # Clean and parse JSON
    try:
        cleaned = clean_json_text(raw)
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise Exception(
            f"Failed to parse MCQ JSON from model.\n"
            f"Raw model output (truncated): {raw[:1000]}\n"
            f"Cleaned output (truncated): {cleaned[:500]}\n"
            f"Error: {e}"
        )

    # Basic validation
    if not isinstance(parsed, list) or len(parsed) < 1:
        raise Exception("MCQ JSON not in expected list format")

    # Limit to exactly 10
    return parsed[:10]

# ------------------------
# Streamlit UI & app logic
# ------------------------
st.set_page_config(page_title="Study + Files RAG (Gemini + Chroma) — Enhanced", layout="wide")

# Custom style
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(120deg, #f6f9ff 0%, #ffffff 100%); }
    .header { padding: 18px; border-radius: 12px; background: linear-gradient(90deg,#ffffff, #f0f4ff); box-shadow: 0 6px 18px rgba(0,0,0,0.06); }
    .file-card { background: rgba(0,0,0,0.02); padding: 10px; border-radius: 8px; margin:6px 0; }
    .side-box { background: rgba(250,250,250,0.8); padding: 12px; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='header'><h1>📚 StudyBuddy AI</h1><p>Upload PDFs / DOCX / TXT, index them, ask questions, generate summaries or quizzes.</p></div>", unsafe_allow_html=True)

# columns
left, right = st.columns([1, 2])

# Initialize session state for quiz
if 'quiz_questions' not in st.session_state:
    st.session_state['quiz_questions'] = None
if 'quiz_index' not in st.session_state:
    st.session_state['quiz_index'] = 0
if 'quiz_score' not in st.session_state:
    st.session_state['quiz_score'] = 0
if 'quiz_done' not in st.session_state:
    st.session_state['quiz_done'] = False
if 'quiz_feedback' not in st.session_state:
    st.session_state['quiz_feedback'] = None
if 'last_query_retrieved' not in st.session_state:
    st.session_state['last_query_retrieved'] = []

with left:
    st.subheader("1) Upload & Index files")
    uploaded = st.file_uploader("Upload multiple files (pdf, docx, txt, md)", type=["pdf","docx","txt","md"], accept_multiple_files=True)
    save_uploaded = st.checkbox("Save uploaded files on server", value=False)
    index_btn = st.button("Process & Index uploaded files")

    if index_btn:
        if not uploaded:
            st.warning("No files uploaded.")
        else:
            with st.spinner("Extracting, chunking and storing embeddings..."):
                emb_model = load_embedding_model()
                client, collection = get_chroma_client_and_collection()
                docs_to_add = []
                for f in tqdm(uploaded, desc="processing files"):
                    fname, text = extract_text(f)
                    if save_uploaded:
                        os.makedirs("uploaded_files", exist_ok=True)
                        with open(os.path.join("uploaded_files", fname), "wb") as out:
                            out.write(f.getbuffer())
                    chunks = chunk_text(text)
                    for i, ch in enumerate(chunks):
                        doc_id = f"{fname}__chunk_{i}__{uuid.uuid4().hex[:8]}"
                        meta = {"source": fname, "chunk_index": i}
                        docs_to_add.append({"id": doc_id, "text": ch, "metadata": meta})
                if docs_to_add:
                    add_documents_to_chroma(collection, emb_model, docs_to_add)
                    try:
                        client.persist()
                    except Exception:
                        pass
                    st.success(f"Indexed {len(docs_to_add)} chunks into Chroma collection `{COLLECTION_NAME}`.")
                else:
                    st.info("No text extracted from uploaded files.")

    st.markdown("---")
    st.subheader("2) Chroma DB")
    if st.button("Show collection stats"):
        client, collection = get_chroma_client_and_collection()
        try:
            n = collection.count()
        except Exception:
            n = "unknown"
        st.write("Collection:", COLLECTION_NAME)
        st.write("Number of entries (approx):", n)

    st.markdown("---")
    st.subheader("3) Gemini API Key & Model")
    gemini_key = st.text_input("Gemini API key (temporary for this session)", type="password")
    model_choice = st.selectbox("Choose Gemini model (example values)", options=[
        "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.5-pro", "gemini-2.5-flash"
    ], index=1)
    st.caption("Use the exact name available in your account. Do not store long-lived keys here.")

with right:
    st.subheader("Ask & Tools")
    question = st.text_area("Enter your question (ask from uploaded files)", placeholder="E.g., Summarize the main points about 'X' from the uploaded materials.")
    top_k = st.slider("Number of context chunks to retrieve", 1, 12, 6)
    max_tokens = st.slider("Max output tokens (approx)", 64, 1024, 512, step=64)
    run_query = st.button("Ask")

    # New: Summary button (dialog-like)
    st.markdown("### Quick features")
    colA, colB = st.columns(2)
    with colA:
        if st.button("📄 Summary"):
            # show options via expander
            st.session_state['_show_summary_panel'] = True
        else:
            if '_show_summary_panel' not in st.session_state:
                st.session_state['_show_summary_panel'] = False

    with colB:
        if st.button("📝 Create Quiz"):
            st.session_state['_show_quiz_panel'] = True
            # reset quiz state
            st.session_state['quiz_questions'] = None
            st.session_state['quiz_index'] = 0
            st.session_state['quiz_score'] = 0
            st.session_state['quiz_done'] = False
            st.session_state['quiz_feedback'] = None
        else:
            if '_show_quiz_panel' not in st.session_state:
                st.session_state['_show_quiz_panel'] = False

    # Summary panel
    if st.session_state.get('_show_summary_panel', False):
        with st.expander("📄 Generate Summary", expanded=True):
            sum_mode = st.radio("Summary mode:", ["Summarize whole uploaded documents", "Summarize a specific topic"], index=0)
            topic_input = ""
            if sum_mode.startswith("Summarize a specific"):
                topic_input = st.text_input("Topic to summarize (phrase):", placeholder="e.g., 'neural networks', 'murder law' etc.")
            summary_top_k = st.slider("Context chunks to use for summary", 1, 12, 6, key="summary_topk")
            summary_btn = st.button("Generate Summary", key="generate_summary")
            if summary_btn:
                if not gemini_key:
                    st.error("Please enter your Gemini API key on the left first.")
                else:
                    with st.spinner("Retrieving relevant chunks and calling Gemini..."):
                        emb_model = load_embedding_model()
                        client_chroma, collection = get_chroma_client_and_collection()
                        # build a query for retrieval: either topic or broad
                        q = topic_input if topic_input else "summarize documents"
                        retrieved = query_chroma(collection, emb_model, q, top_k=summary_top_k)
                        st.session_state['last_query_retrieved'] = retrieved
                        if not retrieved:
                            st.info("No context found in the index. Make sure you indexed files first.")
                        else:
                            # build a concise summarization prompt
                            if topic_input:
                                prompt_q = f"Summarize the key points about '{topic_input}' from the context. Keep it concise (about 6-10 bullet points). Mention sources for important facts."
                            else:
                                prompt_q = "Provide a concise summary of the provided documents. Use bullets and mention sources when helpful."

                            try:
                                init_genai_client(gemini_key)
                                answer = ask_gemini_with_context(model_choice, prompt_q, retrieved, max_output_tokens=max_tokens)
                                st.markdown("### ✅ Summary")
                                st.write(answer)
                            except Exception as e:
                                st.error(f"GenAI call failed: {e}")

    # Quiz panel
    if st.session_state.get('_show_quiz_panel', False):
        with st.expander("📝 Quiz creator", expanded=True):
            quiz_mode = st.radio("Quiz mode:", ["Create quiz from whole documents", "Create quiz from a specific topic"], index=0)
            quiz_topic = ""
            if quiz_mode.startswith("Create quiz from a specific"):
                quiz_topic = st.text_input("Topic for quiz:", placeholder="e.g., 'contract law', 'chapter 2: basics of ML'")
            quiz_top_k = st.slider("Context chunks for generating questions", 2, 20, 12, key="quiz_topk")
            create_quiz_btn = st.button("Generate 10-question MCQ quiz", key="create_quiz_btn")
            if create_quiz_btn:
                if not gemini_key:
                    st.error("Please enter your Gemini API key on the left first.")
                else:
                    with st.spinner("Retrieving context and generating MCQs... This may take 20-40s depending on model..."):
                        emb_model = load_embedding_model()
                        client_chroma, collection = get_chroma_client_and_collection()
                        q_for_retrieval = quiz_topic if quiz_topic else "generate questions"
                        retrieved = query_chroma(collection, emb_model, q_for_retrieval, top_k=quiz_top_k)
                        st.session_state['last_query_retrieved'] = retrieved
                        if not retrieved:
                            st.info("No context found in the index. Make sure you indexed files first.")
                        else:
                            try:
                                mcqs = generate_mcq_json(model_choice, quiz_topic, retrieved, gemini_key, max_output_tokens=1024)
                                # Validate & normalize each question
                                normalized = []
                                for i, item in enumerate(mcqs):
                                    q_text = item.get('question') or item.get('q') or ""
                                    options = item.get('options') or item.get('choices') or []
                                    # ensure len(options)==4
                                    if len(options) != 4:
                                        # attempt to split or fail gracefully
                                        # if options is a dict, convert values
                                        if isinstance(options, dict):
                                            opts = list(options.values())
                                            options = opts[:4] + ([""]*(4-len(opts)))
                                        else:
                                            # fallback: try to split by newline
                                            if isinstance(item.get('options'), str):
                                                opts = [o.strip() for o in item['options'].splitlines() if o.strip()]
                                                options = (opts + [""]*4)[:4]
                                            else:
                                                options = (options + [""]*4)[:4]
                                    answer_index = item.get('answer_index')
                                    if answer_index is None:
                                        # try answer or correct
                                        ai = item.get('answer') or item.get('correct')
                                        # try to find index by matching text
                                        if isinstance(ai, str):
                                            try:
                                                answer_index = options.index(ai)
                                            except Exception:
                                                answer_index = 0
                                        else:
                                            answer_index = int(ai) if isinstance(ai, int) else 0
                                    explanation = item.get('explanation') or ""
                                    normalized.append({
                                        "question": q_text,
                                        "options": options,
                                        "answer_index": int(answer_index),
                                        "explanation": explanation
                                    })
                                # store in session state
                                st.session_state['quiz_questions'] = normalized[:10]
                                st.session_state['quiz_index'] = 0
                                st.session_state['quiz_score'] = 0
                                st.session_state['quiz_done'] = False
                                st.success("Quiz generated! Scroll down to start taking it.")
                            except Exception as e:
                                st.error(f"Failed to generate MCQs: {e}")

    # If user asked a question via the standard "Ask" button
    if run_query:
        if not question.strip():
            st.error("Please enter a question.")
        else:
            if not gemini_key:
                st.error("Please enter your Gemini API key on the left first.")
            else:
                with st.spinner("Searching and calling Gemini..."):
                    emb_model = load_embedding_model()
                    client_chroma, collection = get_chroma_client_and_collection()
                    retrieved = query_chroma(collection, emb_model, question, top_k=top_k)
                    st.session_state['last_query_retrieved'] = retrieved
                    st.markdown("#### Retrieved context (top results):")
                    for r in retrieved:
                        st.markdown(f"- **Source:** {r.get('metadata', {}).get('source','unknown')} (distance={r.get('distance'):.4f})")
                        st.caption((r['text'][:500] + ("..." if len(r['text'])>500 else "")))
                    try:
                        init_genai_client(gemini_key)
                        answer = ask_gemini_with_context(model_choice, question, retrieved, max_output_tokens=max_tokens)
                        st.markdown("### ✅ Answer")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"GenAI call failed: {e}")

    st.markdown("---")
    st.info("Tip: Index your files first (left side) before asking questions. If you re-upload new files, re-index them.")

# ------------------------
# Quiz runner UI (bottom of page)
# ------------------------
st.markdown("---")
st.header("🔎 Quiz: Take & Review")

if st.session_state.get('quiz_questions'):
    questions = st.session_state['quiz_questions']
    idx = st.session_state['quiz_index']
    total = len(questions)

    if st.session_state.get('quiz_done', False):
        st.success(f"Quiz completed! Your score: {st.session_state['quiz_score']} / {total}")
        if st.button("Restart Quiz"):
            st.session_state['quiz_index'] = 0
            st.session_state['quiz_score'] = 0
            st.session_state['quiz_done'] = False
            st.session_state['quiz_feedback'] = None
    else:
        q = questions[idx]
        st.markdown(f"**Question {idx+1} of {total}:**")
        st.write(q['question'])
        choice_key = f"quiz_choice_{idx}"
        # Provide radio for options
        selected = st.radio("Select an option:", q['options'], key=choice_key)
        submit_key = f"submit_{idx}"
        if st.button("Submit Answer", key=submit_key):
            sel_index = q['options'].index(selected)
            correct_index = int(q.get('answer_index', 0))
            if sel_index == correct_index:
                st.session_state['quiz_feedback'] = {"correct": True, "message": "Correct ✅"}
                st.session_state['quiz_score'] += 1
            else:
                st.session_state['quiz_feedback'] = {
                    "correct": False,
                    "message": f"Wrong ❌  | Correct: Option {correct_index+1}: {q['options'][correct_index]}"
                }
            # show explanation if present
            if q.get('explanation'):
                st.write("Explanation:")
                st.write(q['explanation'])
            st.rerun()  # rerun to show feedback and next button

        # show feedback if exists for this question
        fb = st.session_state.get('quiz_feedback')
        if fb:
            if fb.get('correct'):
                st.success(fb.get('message'))
            else:
                st.error(fb.get('message'))

            # Next / Finish
            if idx + 1 < total:
                if st.button("Next Question"):
                    st.session_state['quiz_index'] += 1
                    st.session_state['quiz_feedback'] = None
                    st.rerun()
            else:
                # finish
                if st.button("Finish Quiz"):
                    st.session_state['quiz_done'] = True
                    st.rerun()

else:
    st.info("No quiz loaded. Use the 'Create Quiz' button in the 'Ask & Tools' section to generate a quiz from your indexed documents.")

# Footer
st.markdown("<hr><p style='text-align:center;'>Built with ❤️ — Gemini (GenAI) + sentence-transformers + ChromaDB — Quiz & Summary features added.</p>", unsafe_allow_html=True)
