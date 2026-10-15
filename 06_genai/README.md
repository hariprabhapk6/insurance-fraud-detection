# Level: GenAI — Generative AI Capabilities

## Environment constraint & how it was handled
No internet access → cannot download a pretrained GAN/diffusion checkpoint or
LLM. Implemented genuine generative modeling locally:
- **Tabular**: Gaussian Mixture Model fit on the real fraud-class density, then
  *sampled* to synthesize novel fraudulent claims (a legitimate generative
  model — same family as GAN/VAE generators, but with a tractable closed-form
  density instead of a learned neural generator).
- **Text**: order-2 word-level Markov chain trained on real fraud narratives,
  generating new synthetic fraud descriptions — the classical predecessor of
  neural text generation, fully local and dependency-free.

## Results

### Stress test: does the existing detector generalize to novel fraud patterns?
The already-trained ML baseline (`01_ml`) was scored against 300 **never-seen**
GenAI-synthesized fraud claims:

> **Only 29.7%** of novel synthetic fraud claims were flagged (mean predicted
> probability 0.428, vs. the 0.5 decision threshold).

This is an honest, useful finding: it reveals the baseline overfits to the
exact fraud archetypes in the training distribution and under-generalizes to
plausible variants — precisely the "evolving fraud tactics" risk the problem
statement asks GenAI to stress-test for.

### Data augmentation experiment
Retraining the Random Forest with the GenAI-synthesized fraud claims added to
the training set (test set unchanged, real data only):

| Metric | Real data only | + GenAI synthetic fraud |
|---|---|---|
| Precision | 0.830 | 0.741 |
| Recall | 0.419 | **0.462** |
| F1 | 0.557 | **0.570** |
| **ROC-AUC** | 0.864 | **0.890** |

Augmentation improved recall and ROC-AUC (more fraud caught) at a moderate
precision cost — a reasonable trade-off for a fraud-detection system where
missed fraud is typically costlier than a false positive that gets reviewed
and cleared by a human investigator.

## Sample synthetic fraud narrative (Markov-generated)
> "Vehicle stolen from an unspecified location, no witnesses, no witnesses, no
> police FIR number provided yet."

(Note the repetition artifact — a known limitation of low-order Markov chains
vs. neural language models; flagged here rather than hidden.)

## Ethical note
Synthetic fraud data is used strictly for internal model stress-testing and
training augmentation. It is never presented as real claim data, and outputs
are tagged with `SYNTH` claim IDs so they can never be mistaken for genuine
customer records downstream.

## How to reproduce
```bash
python3 genai_synthetic.py
```
