# agent_cot_alignment

This repository contains an experimental setup for training and evaluating transformer-based
embedding-models designed to assess the alignment of LLM chains of thought with a given prompt.
Note: This is a toy project and should not be confused for scientifically sound/reviewed research.
Inspiration came from the world of graph classifiers/GNNs, which use mappings of embeddings to
distnguish normal/anomalous graph or node samples.

# Background

These experiments were inspired by certain limitations of LLM-based alignment evaluation as well
as a lack of embedding-based alignment checks specifically for agentic tasks.

## Real-time evaluation and decreased costs

Because embedding-models are relatively small in comparision to many LLMs, they offer the
possibility of a faster, even real-time assessment of agent chains of thought. This also allows
for decreased costs in comparison to LLM as a Judge methods due to the relatively small size of
the model.

## Prompt injection is an unsolved problem

A risk of methods such as LLM-as-a-judge for alignment assessment is that LLMs are inately
vulnerable to receiving instructions from their context or the prompt that may not be aligned with
the task of risk assessment. This could lead to potential poisoning of alignment-assessors, for
example due to inserted instructions from an agent that is aware of the judge.

# Methods

## Dataset

Pre-generated prompt, aligned, and misaligned chain-of-thought pairs are loaded from JSON files
under `data/`. Each example consists of a prompt, an aligned response, and a misaligned response.

## Embedder and Evaluator

Two separate DistilBERT-based models are created from `keras_hub` presets. Both use masked mean
pooling over the token sequence and L2-normalize the output embedding. The embedder is frozen and
used to precompute prompt embeddings. The evaluator is the trainable component.

## Contrastive training

The evaluator is trained with a contrastive (InfoNCE-style) loss. For each batch, prompt, aligned,
and misaligned texts are tokenized (using tail-truncation to preserve the end of the sequence) and
fed through the evaluator. Cosine similarity between prompt and response embeddings is scaled by a
temperature parameter, then passed through a softmax over the positive/negative pairs. The loss
encourages aligned embeddings to be closer to their prompt than misaligned ones.

Training uses 80% of the data, with the remaining 20% held out for evaluation.

## Evaluation

Aligned and misaligned similarity is measured as the cosine similarity between the embedder's prompt
embedding and the evaluator's response embedding — higher values indicate better alignment.
Pairwise variance of evaluator embeddings is also computed to track embedding dispersion before and
after training.