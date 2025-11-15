# Design Notes — Member QA System (Bonus 1)

**Source / reference:** Aurora take-home assignment gist. citeturn0view0

---

## Goal

Build a reliable question-answering API that answers natural-language questions about member data (messages). Must be deployed and publicly accessible. This document summarizes alternative approaches, the chosen architecture (RAG + LLM ensemble), metric tuning plan, fine-tuning recommendations, evaluation strategy, and deployment notes.

---

## Executive summary

After evaluating multiple retrieval and inference strategies (rule-based, BM25, semantic retrieval, LLM inference), the LLM-based approach with a retrieval-augmented generation (RAG) pipeline performs best in practice (see evaluation results where `llm` has the highest UEM). This design focuses on combining a vector store (FAISS), strong retrieval (hybrid BM25 + embeddings), and a small, targeted prompt or fine-tuned LLM for robust, explainable answers.

---

## Alternatives considered

1. **Pure rule-based**

   * Pros: Deterministic, fast, low cost.
   * Cons: Fragile, poor recall for paraphrases, yields many unknowns.

2. **BM25-only retrieval + template**

   * Pros: Fast, interpretable.
   * Cons: Surface-form matching; struggles with semantic variants.

3. **Semantic retrieval (embeddings) + nearest neighbors**

   * Pros: Captures paraphrases and intent.
   * Cons: Requires embeddings infra; precision may vary with short messages.

4. **Pure LLM generation (no retrieval)**

   * Pros: Flexible, can reason across messages.
   * Cons: Hallucinations, higher cost, sensitive to prompt tokens.

5. **RAG (preferred)**

   * Hybrid approach: retrieve relevant documents (embeddings + BM25), pass them to LLM as context, instruct to answer or estimate.
   * Balances precision (retrieval) and reasoning (LM).

---

## Chosen architecture (RAG + LLM ensemble)

### Components

1. **Ingest & indexing**

   * Fetch messages from `/messages` endpoint and normalize timestamps, usernames, and text.
   * Index two stores:

     * **FAISS (or Milvus, Weaviate)** for dense embeddings.
     * **BM25 inverted index** for lexical recall.

2. **Retriever**

   * Query both BM25 and FAISS; merge results (score fusion).
   * Deduplicate and rank top-k candidates (k configurable).

3. **Reranker (optional)**

   * Lightweight cross-encoder (distilBERT) to rerank top-k for final precision.

4. **LLM responder**

   * Prompt uses an Alpaca-style instruction: include timestamps and only the selected messages (or person's messages), ask for succinct answer and mark estimates.
   * Two deployment options:

     * Use hosted large model (Gemini / OpenAI / Anthropic) as a service.
     * Use a fine-tuned small LLM (e.g., Llama 2 / Mistral / Falcon) served from a container (self-host or HF Inference).

5. **Ensemble / Arbiter**

   * Keep the existing rule/bm25/semantic methods as fallbacks and for confidence calibration.
   * Optionally use an LLM-as-judge that inspects method outputs and picks best answer or synthesizes final output.

6. **Metrics & feedback loop**

   * Store UEM (Unified Evaluation Metric) per-response; collect human labels where available to update weights.

---

## UEM integration & metric tuning

We define UEM = α * relevance + β * consistency + γ * completeness, with default weights α=0.5, β=0.3, γ=0.2. The system should:

* Log per-method component scores for every query (relevance, consistency, completeness).
* Allow dynamic re-weighting via configuration or an auto-tuner service.
* Optionally learn weights by maximizing correlation with human judgments (small labeled dataset).

**Tuning procedure**

1. Collect a labeled validation set (200–500 questions) with human-graded correctness.
2. Grid-search α,β,γ (subject to sum=1) optimizing correlation / accuracy.
3. Deploy best weights and monitor drift.

---

## Fine-tuning & model choices

### Short-term (fast-to-deploy)

* Use an instruction-tuned LLM (hosted) and rely on RAG to constrain context.
* Advantages: fast, lower engineering cost.

### Medium-term (higher accuracy)

* Fine-tune a smaller LLM on in-domain dialog/message -> answer pairs.
* Data sources: synthetic QA pairs (generated from messages), human-labeled QA pairs.
* Fine-tuning objectives:

  * Supervised fine-tuning (SFT) with instruction + QA pairs.
  * Optionally RLHF or preference tuning with pairwise comparisons to reduce hallucinations.

### Long-term (best performance)

* Train a retrieval-aware LLM (e.g., Fusion-in-Decoder) or fine-tune with real RAG context so model learns to cite and abstain when unsupported.

---

## Confidence calibration & abstention

* When LLM confidence (or UEM) below threshold, return a cautious response: "I don't see evidence for that in the messages; best estimate is..." and attach used messages.
* Expose a `confidence` field in API responses derived from UEM or an ensemble score.

---

## Data pipeline & infra

* **Ingestion**: scheduled fetch + incremental updates; dedupe by message ID.
* **Preprocessing**: normalize timestamps (ISO8601), canonicalize user names, language detection, and basic NER for person detection.
* **Indexing**: nightly reindex + real-time partial updates.
* **Storage**: small DB for metadata (Postgres) + vector DB for embeddings (FAISS on disk, or managed vector DB).

---

## Evaluation & A/B testing

* Hold out a human-labeled test set. Use UEM and exact-match/precision metrics to compare variants.
* A/B test new retriever/hyperparams live for traffic slices.

---

## Risks & mitigations

* **Hallucination**: mitigate via RAG constraints, conservative prompting, and abstention policy.
* **Privacy**: messages may contain PII; minimize stored PII and follow retention policies.
* **Latency & cost**: hybrid retrieval reduces prompt size; prefer smaller/fine-tuned models for production to lower inference cost.

---

## Deployment plan

1. Containerize the API (FastAPI + Uvicorn).
2. Use a vector DB service or host FAISS in the container with persisted index.
3. Deploy to Render / Fly / AWS ECS / GCP Run. Use autoscaling and a health check endpoint.
4. CI: tests for indexing, retrieval, LLM responses (smoke tests).

---

## Next steps & experiments

* Implement cross-encoder reranker to improve top-k precision.
* Collect a small human-labeled dataset for weight tuning and SFT.
* Experiment with a small fine-tuned Llama/Mistral and compare to hosted LLM cost/latency.

---

## Appendix: Quick prompts

**Alpaca-style answer prompt (RAG):**

```
You are a concise QA assistant. Use only the messages listed below. If the messages do not contain a direct answer, provide a short "Estimate:" and explain the reasoning.

Question: {question}

Messages:
[{ts}] {user}: {text}
...

Answer:
```

---

*Document created automatically; references: Aurora take-home gist.* citeturn0view0
