# agent_cot_alignment

This repository contains an experimental setup for training and evaluating transformer-based
embedding-models designed to assess the alignment of LLM chains of thought with a given prompt.

# Background

These experiments were inspired by certain limitations of LLM-based alignment evaluation as well
as a lack of embedding-based alignment checks specifically for agentic tasks.

## Prompt injection is an unsolved problem

A risk of methods such as LLM-as-a-judge for alignment assessment is that LLMs are inately
vulnerable to receiving instructions from their context or the prompt that may not be aligned with
the task of risk assessment. This could lead to potential poisoning of alignment-assessors.

## Real-time evaluation

Because embedding-models are relatively small in comparision to many LLMs, they offer the
possibility of a faster, even real-time assessment of agent chains of thought.

# Methods

## Dataset generation

Due to a lack of alignment-based datasets focusing on agentic tasks, this repository implements
functions to generate a dataset of prompts, aligned and misaligned chains of thought based on
generated industry role-descriptions. The dataset generation is performed with the model
gpt-oss:20b

## Embedding and Evaluation models

Embeddings of the prompts are performed with a pretrained DistilBERT preprocessor and backbone.
A second DistilBERT-based model is then created and trained to minimize the L2 distance between
the aligned chain of thought and the embedding of the prompt itself, the hypothesis being that
aligned examples will exhibit a lower distance to the query embeddings than misaligned examples.

Training is performed on 80% of the dataset and evaluation on the remaining 20%, with evaluation
consisting of a L2 distance comparison between the aligned and misaligned samples. Given aligned
embeddings that are consistently closer than misaligned embeddings, a hypersphere can be defined
in the embedding space such that it is centered around the prompt embedding and reliable
encompasses the aligned samples while excliding the misaligned samples.