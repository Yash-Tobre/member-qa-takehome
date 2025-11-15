# main.py
from fastapi import FastAPI, Query, HTTPException
from typing import List, Dict, Any, Optional
import requests
import os
from datetime import datetime
from dotenv import load_dotenv
import logging
import csv
import math

# optional heavy imports
try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None
    util = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

import spacy
import os

import spacy
from spacy.util import is_package, get_package_path

model_name = "en_core_web_sm"

nlp = spacy.load("en_core_web_sm")

# google genai SDK optional
try:
    import google.genai as genai
    GEMINI_CLIENT = None
except Exception:
    genai = None
    GEMINI_CLIENT = None

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Member QA — multi-method with timestamp (Gemini API)")

MESSAGES_API = os.environ.get(
    "MESSAGES_API",
    "https://november7-730026606190.europe-west1.run.app/messages?limit=1000"
)

GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")

INTENT_VOCAB = {
    "travel": ["trip", "travel", "flight", "plane", "journey", "vacation", "holiday", "flights", "going to", "book", "hotel"],
    "food": ["restaurant", "dinner", "lunch", "food", "eat", "meal", "reservation", "table"],
    "vehicle": ["car", "cars", "vehicle", "drive", "owned", "ownership", "license", "parking"]
}

_model_cache = {
    "embedder": None,
    "bm25": None,
    "bm25_corpus": None,
    "llm_model": None
}

# ---------------------------
# Utilities
# ---------------------------
def fetch_messages_debug():
    resp = requests.get(MESSAGES_API, timeout=15)
    status = resp.status_code
    text_head = (resp.text or "")[:1000]
    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(f"Messages API did not return JSON (status {status}).\nRaw response head:\n{text_head}")
    return data

def fetch_messages() -> List[Dict[str, Any]]:
    data = fetch_messages_debug()
    if isinstance(data, dict) and "items" in data:
        items = data["items"]
    elif isinstance(data, list):
        items = data
    else:
        raise RuntimeError("Unexpected messages JSON structure (no 'items' list).")
    # parse timestamps
    for m in items:
        ts = m.get("timestamp")
        if ts:
            try:
                m["_dt"] = datetime.fromisoformat(ts)
            except Exception:
                m["_dt"] = None
        else:
            m["_dt"] = None
    return sorted(items, key=lambda x: x["_dt"] or datetime.min)

def detect_person(question: str, messages: List[Dict[str, Any]]) -> Optional[str]:
    if nlp:
        doc = nlp(question)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
    ql = (question or "").lower()
    for m in messages:
        name = m.get("user_name","")
        if not name:
            continue
        for token in name.split():
            if token.lower() in ql and len(token) > 2:
                return name
    return None

def format_msg(m: Dict[str,Any]) -> str:
    ts = m.get("_dt")
    ts_str = ts.isoformat() if ts else m.get("timestamp") or "unknown time"
    return f"[{ts_str}] {m.get('user_name','')}: {m.get('message','')}"

# ---------------------------
# Rule-based
# ---------------------------
def method_rule(question: str, person: Optional[str], messages: List[Dict[str,Any]]):
    ql = (question or "").lower()
    intent = None
    for cat, words in INTENT_VOCAB.items():
        if any(w in ql for w in words):
            intent = cat
            break
    pool = [m for m in messages if not person or person.lower() in m.get("user_name","").lower()]
    matches = []
    if intent:
        keywords = INTENT_VOCAB[intent]
        for m in pool:
            text = (m.get("message") or "").lower()
            if any(k in text for k in keywords):
                matches.append(m)
    if matches:
        best = sorted(matches, key=lambda x: x["_dt"] or datetime.min, reverse=True)[0]
        return {"method":"rule", "answer": best.get("message"), "picked": format_msg(best)}
    if pool:
        recent = sorted(pool, key=lambda x: x["_dt"] or datetime.min, reverse=True)[0]
        return {"method":"rule", "answer": recent.get("message"), "picked": format_msg(recent)}
    return {"method":"rule", "answer": None, "picked": None}

# ---------------------------
# Timestamp-aware
# ---------------------------
def method_timestamp(question: str, person: Optional[str], messages: List[Dict[str,Any]]):
    ql = (question or "").lower()
    pool = [m for m in messages if not person or person.lower() in m.get("user_name","").lower()]
    if not pool:
        return {"method":"timestamp", "answer": None, "timestamp": None}
    intent = None
    for cat, words in INTENT_VOCAB.items():
        if any(w in ql for w in words):
            intent = cat
            break
    matches = []
    if intent:
        keywords = INTENT_VOCAB[intent]
        for m in pool:
            text = (m.get("message") or "").lower()
            if any(k in text for k in keywords):
                matches.append(m)
    best = sorted(matches or pool, key=lambda x: x["_dt"] or datetime.min, reverse=True)[0]
    return {"method":"timestamp", "answer": best.get("message"), "timestamp": best.get("_dt").isoformat() if best.get("_dt") else None}

# ---------------------------
# BM25
# ---------------------------
def ensure_bm25(messages: List[Dict[str,Any]]):
    if BM25Okapi is None:
        raise RuntimeError("BM25 library missing. pip install rank_bm25")
    if _model_cache.get("bm25") is not None and _model_cache.get("bm25_corpus") == messages:
        return _model_cache["bm25"]
    corpus = [(m.get("message") or "").lower().split() for m in messages]
    bm25 = BM25Okapi(corpus)
    _model_cache["bm25"] = bm25
    _model_cache["bm25_corpus"] = messages
    return bm25

def method_bm25(question: str, person: Optional[str], messages: List[Dict[str,Any]], top_k:int=3):
    pool = [m for m in messages if not person or person.lower() in m.get("user_name","").lower()]
    if not pool:
        return {"method":"bm25", "answer": None, "hits":[]}
    bm25 = ensure_bm25(pool)
    q_tokens = (question or "").lower().split()
    scores = bm25.get_scores(q_tokens)
    ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[:top_k]
    hits = [{"score": float(scores[idx]), "message": pool[idx].get("message"), "picked": format_msg(pool[idx])} for idx, _ in ranked]
    best = hits[0] if hits else None
    return {"method":"bm25", "answer": best["message"] if best else None, "picked": best["picked"] if best else None, "hits": hits}

# ---------------------------
# Semantic
# ---------------------------
def ensure_embedder():
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers missing. pip install sentence-transformers")
    if _model_cache["embedder"] is None:
        _model_cache["embedder"] = SentenceTransformer("all-MiniLM-L6-v2")
    return _model_cache["embedder"]

def method_semantic(question: str, person: Optional[str], messages: List[Dict[str,Any]], top_k:int=3):
    pool = [m for m in messages if not person or person.lower() in m.get("user_name","").lower()]
    if not pool:
        return {"method":"semantic", "answer": None, "hits":[]}
    try:
        embedder = ensure_embedder()
        docs = [m.get("message") or "" for m in pool]
        doc_embs = embedder.encode(docs, convert_to_tensor=True)
        q_emb = embedder.encode([question], convert_to_tensor=True)
        sims = util.pytorch_cos_sim(q_emb, doc_embs)[0].cpu().numpy()
        ranked_idx = sims.argsort()[::-1][:top_k]
        hits = [{"score": float(sims[i]), "message": pool[i].get("message"), "picked": format_msg(pool[i])} for i in ranked_idx]
        best = hits[0] if hits else None
        return {"method":"semantic", "answer": best["message"] if best else None, "picked": best["picked"] if best else None, "hits": hits}
    except Exception as e:
        # graceful fallback if embedder missing
        logging.warning(f"Semantic method failed: {e}")
        # naive heuristic: use bm25
        try:
            return method_bm25(question, person, messages, top_k=top_k)
        except Exception:
            return {"method":"semantic", "answer": None, "hits": [], "error": str(e)}

# ---------------------------
# LLM (Alpaca-style + Gemini fallback)
# ---------------------------
def ensure_llm():
    global GEMINI_CLIENT
    if GEMINI_CLIENT is not None:
        return GEMINI_CLIENT
    if genai is None:
        raise RuntimeError("google-genai SDK not installed. Run: pip install google-genai")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment variables.")
    GEMINI_CLIENT = genai.Client(api_key=api_key)
    logging.info("Gemini client initialized.")
    return GEMINI_CLIENT

def method_llm(question, person, messages, top_k=10):
    """
    Alpaca-style LLM reasoning:
    - If person detected → use only their messages
    - Otherwise → use top 10 most relevant messages overall
    - Always return an answer (estimated if needed)
    """
    # 1. select pool
    if person:
        pool = [m for m in messages if person.lower() in m.get("user_name","").lower()]
    else:
        pool = messages

    if not pool:
        pool = messages

    # 2. rank with semantic, fallback to bm25
    sem = method_semantic(question, None, pool, top_k=top_k)
    hits = sem.get("hits", []) if isinstance(sem, dict) else []
    if not hits:
        try:
            bm = method_bm25(question, None, pool, top_k=top_k)
            hits = bm.get("hits", []) if isinstance(bm, dict) else []
        except Exception:
            hits = []

    # build relevant blocks
    relevant_blocks = []
    for h in hits[:top_k]:
        raw = h.get("message")
        msg_obj = next((m for m in pool if m.get("message") == raw), None)
        ts = msg_obj.get("_dt").isoformat() if msg_obj and msg_obj.get("_dt") else "unknown"
        user = msg_obj.get("user_name", "unknown") if msg_obj else "unknown"
        relevant_blocks.append(f"[{ts}] {user}: {raw}")

    relevant_text = "\n".join(relevant_blocks) if relevant_blocks else "None available."

    prompt = f"""
### Instruction:
You are an inference-focused reasoning model. Use the user's question and only the historical messages provided below to answer.
Do NOT repeat the messages. Provide a single, concise answer or estimate.
If a person is detected, use only their messages. If no person is detected, use the top {top_k} relevant messages.
If exact information is unavailable, provide a best-effort estimate and start with the word "Estimate:".

### User Question:
{question}

### Relevant Historical Messages:
{relevant_text}

### Task:
Provide the most likely and reasonable answer in one sentence. Do not quote or repeat the messages.

### Answer:
""".strip()


    # call Gemini if available, otherwise produce a heuristic fallback
    try:
        client = ensure_llm()
        response = client.models.generate_content(model=GEMINI_MODEL_NAME, contents=prompt,  temperature=0.0 )
        answer = (response.text or "").strip()
        if not answer:
            answer = "Estimate: No direct answer found; based on messages the best estimate is X."
        return {
            "method": "llm",
            "answer": answer,
            "used_messages": relevant_blocks,
            "person_detected": person
        }
    except Exception as e:
        # fallback heuristic: synthesize a short estimate using the top hit(s)
        logging.warning(f"LLM call failed or not available: {e}")
        if relevant_blocks:
            synthesized = "Estimate: Based on these messages — " + " | ".join(relevant_blocks[:3])
            return {"method":"llm", "answer": synthesized, "used_messages": relevant_blocks, "person_detected": person, "error": str(e)}
        else:
            return {"method":"llm", "answer": "Estimate: No data available to infer a precise answer.", "used_messages": [], "person_detected": person, "error": str(e)}

# ---------------------------
# API endpoints
# ---------------------------
@app.get("/ask")
def ask_minimal(question: str = Query(..., description="User question")):
    """
    Minimal endpoint:
    - runs LLM inference only
    - returns the final answer only
    """
    messages = fetch_messages()
    person = detect_person(question, messages)

    try:
        llm_result = method_llm(question, person, messages)
        answer = llm_result.get("answer") or "Unknown"
        return {"answer": answer}
    except Exception as e:
        return {"answer": "Unknown", "error": str(e)}

@app.get("/ask_llm")
def ask_llm(question: str = Query(...)):
    messages = fetch_messages()
    person = detect_person(question, messages)
    llm_res = method_llm(question, person, messages, top_k=10)
    return {"question": question, "person_detected": person, "llm": llm_res}

@app.get("/ask_methods")
def ask_methods(question: str = Query(...), top_k: int = 3):
    messages = fetch_messages()
    person = detect_person(question, messages)
    results = {
        "rule": method_rule(question, person, messages),
        "timestamp": method_timestamp(question, person, messages),
    }
    try:
        results["bm25"] = method_bm25(question, person, messages, top_k=top_k)
    except Exception as e:
        results["bm25"] = {"method":"bm25", "answer": None, "error": str(e)}
    try:
        results["semantic"] = method_semantic(question, person, messages, top_k=top_k)
    except Exception as e:
        results["semantic"] = {"method":"semantic", "answer": None, "error": str(e)}
    try:
        results["llm"] = method_llm(question, person, messages, top_k=10)
    except Exception as e:
        results["llm"] = {"method":"llm", "answer": None, "error": str(e)}
    # return concise answers
    concise = {k: (v.get("answer") if isinstance(v, dict) else v) for k,v in results.items()}
    return {"question": question, "person_detected": person, "methods": concise, "raw": results}

@app.get("/ask_full")
def ask_full(question: str = Query(...), top_k: int = 3):
    messages = fetch_messages()
    person = detect_person(question, messages)
    results = {
        "rule": method_rule(question, person, messages),
        "timestamp": method_timestamp(question, person, messages),
    }
    try:
        results["bm25"] = method_bm25(question, person, messages, top_k=top_k)
    except Exception as e:
        results["bm25"] = {"method":"bm25", "answer": None, "error": str(e)}
    try:
        results["semantic"] = method_semantic(question, person, messages, top_k=top_k)
    except Exception as e:
        results["semantic"] = {"method":"semantic", "answer": None, "error": str(e)}
    try:
        results["llm"] = method_llm(question, person, messages, top_k=10)
    except Exception as e:
        results["llm"] = {"method":"llm", "answer": None, "error": str(e)}
    return {"question": question, "person_detected": person, "results": results}

# ---------------------------
# Evaluation logic (hardcoded 10 questions)
# ---------------------------
SAMPLE_QUESTIONS = [
    "What car does John drive?",
    "Did Sarah travel to Europe last year?",
    "Has Alex ever booked a hotel through the company?",
    "When did Maria last take vacation?",
    "Who usually handles the dinner reservations?",
    "Has anyone mentioned parking issues recently?",
    "Is there a record of flights being booked for June?",
    "Did Sam say anything about buying a new vehicle?",
    "Has the team talked about holiday plans?",
    "Who confirmed the restaurant booking on Friday?"
]

def tokenize_simple(text: str):
    return set((text or "").lower().split())

def jaccard(a:set, b:set):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0

def compute_embedding_similarity(a: str, b: str):
    # preferred path: sentence-transformers
    if SentenceTransformer is not None:
        try:
            embedder = ensure_embedder()
            embs = embedder.encode([a,b], convert_to_tensor=True)
            score = util.pytorch_cos_sim(embs[0], embs[1]).item()
            # convert [-1,1] to [0,1]
            return max(0.0, min(1.0, (score + 1) / 2))
        except Exception as e:
            logging.warning(f"Embedding similarity failed: {e}")
    # fallback: jaccard on tokens
    return jaccard(tokenize_simple(a), tokenize_simple(b))

@app.get("/evaluate_methods")
def evaluate_methods():
    messages = fetch_messages()
    results_rows = []
    # iterate questions
    for q in SAMPLE_QUESTIONS:
        person = detect_person(q, messages)
        # gather outputs
        m_rule = method_rule(q, person, messages)
        m_timestamp = method_timestamp(q, person, messages)
        try:
            m_bm25 = method_bm25(q, person, messages, top_k=3)
        except Exception as e:
            m_bm25 = {"method":"bm25", "answer": None, "error": str(e)}
        try:
            m_sem = method_semantic(q, person, messages, top_k=3)
        except Exception as e:
            m_sem = {"method":"semantic", "answer": None, "error": str(e)}
        try:
            m_llm = method_llm(q, person, messages, top_k=10)
        except Exception as e:
            m_llm = {"method":"llm", "answer": None, "error": str(e)}
        # collect answers
        method_answers = {
            "rule": m_rule.get("answer"),
            "timestamp": m_timestamp.get("answer"),
            "bm25": m_bm25.get("answer") if isinstance(m_bm25, dict) else None,
            "semantic": m_sem.get("answer") if isinstance(m_sem, dict) else None,
            "llm": m_llm.get("answer") if isinstance(m_llm, dict) else None
        }
        # compute pairwise similarities
        answers_list = list(method_answers.items())
        embeddings_cache = {}
        # compute relevance: similarity between method answer and llm answer
        llm_ans = method_answers.get("llm") or ""
        for name, ans in answers_list:
            ans_text = ans or ""
            relevance = compute_embedding_similarity(ans_text, llm_ans) if llm_ans else 0.0
            # consistency: avg similarity between this method answer and all other method answers
            sims = []
            for other_name, other_ans in answers_list:
                if other_name == name:
                    continue
                sims.append(compute_embedding_similarity(ans_text, other_ans or ""))
            consistency = float(sum(sims) / len(sims)) if sims else 0.0
            # completeness: simple heuristic based on token length (caps at 1.0)
            length_tokens = len(tokenize_simple(ans_text))
            completeness = min(1.0, length_tokens / 20.0)  # >=20 tokens -> 1.0
            # final score: weighted sum (relevance 50%, consistency 30%, completeness 20%)
            final_score = 0.5 * relevance + 0.3 * consistency + 0.2 * completeness
            row = {
                "question": q,
                "person_detected": person or "",
                "method": name,
                "answer": ans_text,
                "relevance": round(relevance, 4),
                "consistency": round(consistency, 4),
                "completeness": round(completeness, 4),
                "final_score": round(final_score, 4)
            }
            results_rows.append(row)

    # write CSV
    csv_path = os.path.join(os.getcwd(), "evaluation_results.csv")
    fieldnames = ["question", "person_detected", "method", "answer", "relevance", "consistency", "completeness", "final_score"]
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results_rows:
                writer.writerow(r)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write CSV: {e}")

    # return small preview (first 10 rows)
    preview = results_rows[:10]
    return {"status": "ok", "csv_path": csv_path, "rows_written": len(results_rows), "preview": preview}

# ---------------------------
# shutdown
# ---------------------------
@app.on_event("shutdown")
async def shutdown_event():
    global GEMINI_CLIENT
    if GEMINI_CLIENT:
        try:
            await GEMINI_CLIENT.aclose()
        except AttributeError as e:
            logging.warning("Suppressed AttributeError during Gemini client shutdown.")
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")

#if __name__ == "__main__":
    #import uvicorn
    #uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT",8000)), reload=True)
