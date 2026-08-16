# Hierarchical Graph Attention Networks for Long-Context Document Summarization

Does explicitly modeling a long document's hierarchical structure (sentence → paragraph
→ section → document) as a graph, processed with a Graph Attention Network, improve
abstractive summarization over a flat Transformer baseline — and is it more efficient?

This repository contains the implementation, experiments, and results for a controlled
comparison between a flat T5-small summarizer and a Hierarchical GAT-based summarizer on
the PubMed long-document summarization dataset.

## Headline Result

The flat baseline outperforms the Hierarchical GAT model on summarization quality at the
training scale tested here (983 documents, 3 epochs) — primarily because the baseline
reuses T5's pretrained encoder, while the GAT model must teach the decoder to interpret
an entirely new representation from scratch. However, **within** the graph-based
approach, the attention mechanism itself matters enormously: removing it (GAT → SAGEConv)
roughly halves ROUGE-1. See `experiments.md` for full results and analysis.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Baseline (flat T5-small) | 0.2280 | 0.0912 | 0.1649 |
| Hierarchical GAT (full) | 0.1814 | 0.0252 | 0.1107 |
| Ablation: no semantic edges | 0.1646 | 0.0242 | 0.1037 |
| Ablation: no attention (SAGEConv) | 0.0812 | 0.0030 | 0.0642 |

## Dataset

[`ccdv/pubmed-summarization`](https://huggingface.co/datasets/ccdv/pubmed-summarization)
— a script-free Hugging Face mirror of the PubMed long-document summarization corpus. The
original `scientific_papers` loader is deprecated as of current `datasets` library
versions and is not used here.

## Setup

```bash
pip install torch-geometric transformers datasets evaluate rouge-score bert-score \
    spacy sentence-transformers --break-system-packages
```


Developed and run on Google Colab with a T4 GPU.

## Reproducing the Results

1. Run the preprocessing + graph construction pipeline (`src/graph/graph_dataset.py`) on
   the training split.
2. Train the baseline: `src/training/trainer.py --model baseline`
3. Train the Hierarchical GAT model: `src/training/trainer.py --model hgat`
4. Evaluate both: `src/evaluation/evaluate.py`

See `methodology.md` for full architectural and training details, and `experiments.md`
for the complete results, ablations, and a full narrative of bugs found and fixed during
development.

## Known Limitations

- BERTScore could not be computed due to a `bert_score`/`transformers` version
  incompatibility in the development environment; ROUGE was used as the sole automatic
  quality metric.
- Graph batching (PyG `DataLoader`) was not implemented; the Hierarchical GAT model
  trains with batch size 1.
- Three of five planned ablations (paragraph-node removal, section-node removal, pooling
  strategy) were not completed due to time constraints.

Full details in `experiments.md`.
