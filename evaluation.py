# evaluation.py
import os
import csv
import logging
from typing import List, Dict

# lazy imports to avoid circular dependency
def import_main_funcs():
    from main import fetch_messages, detect_person, method_rule, method_timestamp, method_bm25, method_semantic, method_llm
    from main import tokenize_simple, compute_embedding_similarity
    return fetch_messages, detect_person, method_rule, method_timestamp, method_bm25, method_semantic, method_llm, tokenize_simple, compute_embedding_similarity

def generate_dynamic_questions(messages: List[Dict]) -> List[str]:
    """
    Generate evaluation questions based on actual user_names.
    """
    user_names = list({m.get("user_name") for m in messages if m.get("user_name")})
    questions = []
    for name in user_names:
        questions.append(f"What messages did {name} send recently?")
        questions.append(f"Has {name} talked about travel?")
        questions.append(f"Has {name} mentioned food or dinner?")
        questions.append(f"Has {name} discussed vehicles or parking?")
    return questions

def evaluate_methods_dynamic():
    fetch_messages, detect_person, method_rule, method_timestamp, method_bm25, method_semantic, method_llm, tokenize_simple, compute_embedding_similarity = import_main_funcs()
    
    messages = fetch_messages()
    SAMPLE_QUESTIONS = generate_dynamic_questions(messages)
    results_rows = []

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
        llm_ans = method_answers.get("llm") or ""
        for name, ans in answers_list:
            ans_text = ans or ""
            relevance = compute_embedding_similarity(ans_text, llm_ans) if llm_ans else 0.0
            sims = [compute_embedding_similarity(ans_text, other_ans or "") 
                    for other_name, other_ans in answers_list if other_name != name]
            consistency = float(sum(sims) / len(sims)) if sims else 0.0
            completeness = min(1.0, len(tokenize_simple(ans_text)) / 20.0)
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
    csv_path = os.path.join(os.getcwd(), "evaluation_results_dynamic.csv")
    fieldnames = ["question", "person_detected", "method", "answer", "relevance", "consistency", "completeness", "final_score"]
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results_rows:
                writer.writerow(r)
    except Exception as e:
        logging.error(f"Failed to write CSV: {e}")
        raise

    return {"status": "ok", "csv_path": csv_path, "rows_written": len(results_rows), "preview": results_rows[:10]}

if __name__ == "__main__":
    result = evaluate_methods_dynamic()
    print(result)
