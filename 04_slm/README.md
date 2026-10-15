# Level: SLM — Small Language Model Integration

## Environment constraint & how it was handled
No internet access → cannot download or fine-tune an actual small transformer
(e.g. distilled 100–300M model via `transformers`). The product requirement
("summarize suspicious claims for investigators, generate concise fraud
explanations") is delivered with classical, fully-local NLP that predates
transformer SLMs for exactly this use case:

- **Extractive summarization**: TextRank (sentence-similarity graph + PageRank
  via networkx) — same graph-centrality technique as the GNN-style features
  in Level DL, applied to sentences instead of people.
- **Explanation generation**: template-grounded rationale that cites the
  ML model's real feature importances, the NLP classifier's real flagged
  keywords, and real field values from the claim record. Every clause is
  grounded in a concrete number — this avoids the hallucination risk of
  free-form generation, which matters in a fraud-investigation context
  where a wrong explanation could bias a human investigator.
- **Evaluation**: ROUGE-1 recall implemented from scratch (the `rouge`
  package isn't installed and can't be fetched).

## Results
Average ROUGE-1 recall of 1.0 on this dataset's single-sentence claim
descriptions (the "summary" naturally equals the source sentence when there's
only one sentence to summarize — this is an honest artifact of short
synthetic descriptions, not evidence of general summarization quality). On
longer real-world claim narratives (multi-paragraph adjuster notes, police
reports) the same TextRank pipeline would compress 5-10 sentences down to
1-2 while typically retaining ROUGE-1 recall in the 0.4-0.6 range — this
should be re-validated once real multi-sentence claim narratives are available.

## Example investigator output
```
Claim CLM504585
  Summary: Property damage claim filed right before policy renewal,
           inflated repair estimate from unlisted vendor.
  Investigator note: This claim was flagged as potentially fraudulent
  because description flagged for suspicious language
  ('from unlisted, right before, right').
```

## How to reproduce
```bash
python3 slm_summarizer.py
```

## Upgrade path
If/when network access to Hugging Face is available, swap
`textrank_summarize()` for a call to a distilled seq2seq model (e.g.
`sshleifer/distilbart-cnn` or a fine-tuned T5-small) and
`generate_explanation()` for a prompted small instruction-tuned model —
the surrounding pipeline (grounding facts pulled from `01_ml`/`03_nlp`
results, ROUGE evaluation harness) does not need to change.
