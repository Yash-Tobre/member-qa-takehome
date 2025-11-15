# Member QA Evaluation Results

This document summarizes the evaluation of various QA methods applied to our member messages dataset. The evaluation used a **Unified Evaluation Metric (UEM)** combining relevance, consistency, and completeness.

---

## Evaluation Methodology

We used the following metrics for each method's answer:

1. **Relevance (50%)** – How closely the method's answer matches the LLM-generated answer, computed via embedding similarity or token overlap.
2. **Consistency (30%)** – How consistent the answer is with other methods’ answers for the same question.
3. **Completeness (20%)** – Simple heuristic based on answer length (number of tokens), capped at 20 tokens.

The **Unified Evaluation Metric (UEM)** is computed as:

***UEM = 0.50 * relevance + 0.30 * consistency + 0.20 * completeness***


---

## Methods Evaluated

The following methods were compared:

- **LLM** – Language model inference using historical messages.
- **Semantic** – Sentence embedding-based similarity.
- **Rule** – Simple keyword-based rule method.
- **Timestamp** – Retrieves most recent relevant message.
- **BM25** – Traditional BM25 ranking on tokenized messages.

---

## Results

Average UEM scores per method:

| Method    | UEM Score |
|-----------|-----------|
| LLM       | 0.771     |
| Semantic  | 0.646     |
| Rule      | 0.627     |
| Timestamp | 0.627     |
| BM25      | 0.579     |

**Interpretation:**

- **LLM** outperforms all other methods, showing the highest alignment with relevance and consistency across answers.
- **Semantic** embeddings provide a strong second-best approach.
- **Rule** and **Timestamp** methods perform similarly, offering moderate accuracy.
- **BM25** scores lowest, indicating keyword-only approaches are less effective for this dataset.

---

## Conclusion

The evaluation demonstrates that leveraging LLMs with historical message context provides the most accurate and consistent answers. Semantic embeddings are a strong alternative, while simple rule-based or BM25 methods are less effective.
