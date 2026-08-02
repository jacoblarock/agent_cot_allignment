from typing import Callable
import keras
import tensorflow as tf
import numpy as np


def capture_token_embeddings(
    model: tuple[Callable,keras.Model],
    x: list[str],
) -> list[tf.Tensor]:
    p, m = model
    tokens = p(x)
    backbone = m.get_layer("distil_bert_backbone")
    embeddings = backbone(tokens)
    mask = tokens["padding_mask"]
    return [tf.boolean_mask(embeddings[i], mask[i]) for i in range(len(x))]


def cosine_similarity_to_reference(
    embeddings: tf.Tensor,
    reference: tf.Tensor,
) -> np.ndarray:
    embeddings = tf.cast(embeddings, tf.float32)
    reference = tf.cast(reference, tf.float32)
    reference = tf.reshape(reference, [1, -1])
    embeddings_norm = tf.nn.l2_normalize(embeddings, axis=1)
    reference_norm = tf.nn.l2_normalize(reference, axis=1)
    sims = tf.reduce_sum(embeddings_norm * reference_norm, axis=1)
    return sims.numpy()


def token_similarity_to_reference(
    model: tuple[Callable, keras.Model],
    text: str,
    reference: tf.Tensor,
) -> list[dict]:
    p, m = model
    preprocessor = p.args[0] if hasattr(p, "args") else None
    tokenizer = preprocessor.tokenizer if preprocessor is not None else None
    tokens = p([text])
    token_ids = tokens["token_ids"][0]
    mask = tokens["padding_mask"][0]
    backbone = m.get_layer("distil_bert_backbone")
    embeddings = backbone(tokens)[0]
    masked_embeddings = tf.boolean_mask(embeddings, mask)
    masked_ids = tf.boolean_mask(token_ids, mask)
    sims = cosine_similarity_to_reference(masked_embeddings, reference)
    if tokenizer is not None:
        ids_2d = tf.reshape(masked_ids, [-1, 1])
        token_strings = tokenizer.detokenize(ids_2d)
    else:
        token_strings = masked_ids
    return [
        {"token": str(t), "sim": float(s)}
        for t, s in zip(token_strings, sims)
    ]