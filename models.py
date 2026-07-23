import keras
import keras_hub
import numpy as np
import tensorflow as tf
from functools import partial
from typing import Callable
from scipy.spatial.distance import cdist

losses_map = {
    "mean_absolute_error": keras.losses.mean_absolute_error,
    "mean_squared_error": keras.losses.mean_squared_error,
    "cosine_similarity": keras.losses.cosine_similarity,
}

def tail_preprocessor(preprocessor: keras_hub.models.DistilBertPreprocessor, sequence_length: int, x: list[str]) -> dict[str,tf.Tensor]:
    tokens = tf.ragged.constant(preprocessor.tokenizer(x))
    tokens = tokens[:, -(sequence_length-2):]
    batch_size = tf.shape(tokens)[0]
    start_col = tf.fill([batch_size, 1], 101)
    end_col = tf.fill([batch_size, 1], 102)
    result = tf.concat([start_col, tokens, end_col], axis=1).to_tensor(default_value=0, shape=[None, sequence_length])
    mask = result > 0
    return {
        "token_ids": result,
        "padding_mask": mask,
    }

def masked_mean(args):
    embeddings, mask = args
    mask = keras.ops.expand_dims(keras.ops.cast(mask, embeddings.dtype), axis=-1)
    masked_embeddings = embeddings * mask
    sum_embeddings = keras.ops.sum(masked_embeddings, axis=1)
    seq_len = keras.ops.sum(keras.ops.cast(mask, "float32"), axis=1)
    return sum_embeddings / keras.ops.maximum(seq_len, 1e-8)

def create_model(sequence_length: int, loss: str) -> tuple[Callable, keras.Model]:
    loss_func = losses_map[loss]
    preprocessor = keras_hub.models.DistilBertPreprocessor.from_preset(
        "distil_bert_base_en",
        sequence_length=sequence_length,
    )
    backbone = keras_hub.models.DistilBertBackbone.from_preset(
        "distil_bert_base_en_uncased"
    )
    inputs = backbone.input 
    x = backbone(inputs)
    mask = inputs["padding_mask"]
    outputs = keras.layers.Lambda(masked_mean)([x, mask])
    outputs = keras.layers.Lambda(lambda e: e / keras.ops.sqrt(keras.ops.maximum(keras.ops.sum(e**2, axis=1, keepdims=True), 1e-12)), output_shape=lambda s: s)(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-5),
        loss=loss_func,
    )
    return partial(tail_preprocessor, preprocessor, sequence_length), model

def predict(
    model: tuple[Callable,keras.Model],
    x: list[str],
) -> np.ndarray:
    p, m = model
    return m.predict(p(x), batch_size=32)

def fit(
    model: tuple[Callable,keras.Model],
    x: list[str],
    y: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32
) -> tuple[Callable,keras.Model]:
    p, m = model
    preprocessed = p(x)
    m.fit(preprocessed, y, epochs=epochs, batch_size=batch_size)
    return p, m

def make_batches(
    p_train: list[str], a_train: list[str], m_train: list[str],
    p_embeds: np.ndarray, batch_size: int,
):
    n = len(p_train)
    indices = np.random.permutation(n)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield (
            [p_train[i] for i in batch_idx],
            [a_train[i] for i in batch_idx],
            [m_train[i] for i in batch_idx],
            p_embeds[batch_idx],
        )

def train_evaluator_contrastive(
    evaluator: tuple[Callable, keras.Model],
    p_embeds: np.ndarray,
    p_train: list[str],
    a_train: list[str],
    m_train: list[str],
    epochs: int = 30,
    batch_size: int = 4,
    temperature: float = 0.1,
):
    e_preprocessor, e_model = evaluator
    optimizer = keras.optimizers.Adam(learning_rate=2e-5)

    @tf.function
    def train_step(a_tokens, m_tokens, p_embeds_t):
        with tf.GradientTape() as tape:
            a_embeds = e_model(a_tokens)
            m_embeds = e_model(m_tokens)

            pos_sim = tf.reduce_sum(p_embeds_t * a_embeds, axis=-1) / temperature
            neg_sim = tf.reduce_sum(p_embeds_t * m_embeds, axis=-1) / temperature

            pos_exp = tf.exp(pos_sim)
            neg_exp = tf.exp(neg_sim)

            loss = -tf.reduce_mean(tf.math.log(pos_exp / (pos_exp + neg_exp)))

        grads = tape.gradient(loss, e_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, e_model.trainable_weights))
        return loss

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch_p, batch_a, batch_m, batch_p_emb in make_batches(
            p_train, a_train, m_train, p_embeds, batch_size,
        ):
            a_tokens = e_preprocessor(batch_a)
            m_tokens = e_preprocessor(batch_m)
            p_embeds_t = tf.constant(batch_p_emb, dtype=tf.float32)

            loss = train_step(a_tokens, m_tokens, p_embeds_t)

            total_loss += float(loss)
            n_batches += 1

        print(f"Epoch {epoch+1}/{epochs} - loss: {total_loss / max(n_batches, 1):.4f}")

def eval(
    embedder: tuple[Callable,keras.Model],
    evaluator: tuple[Callable,keras.Model],
    x: list[str],
    y: list[str],
) -> np.ndarray:
    x_embed = predict(embedder, x)
    y_embed = predict(evaluator, y)
    return keras.ops.sum(x_embed * y_embed, axis=-1).numpy()

def get_variances(
    model: tuple[Callable,keras.Model],
    x: list[str],
) -> np.ndarray:
    x_embed = predict(model, x)
    return cdist(x_embed, x_embed, metric="euclidean")

def find_best_threshold(
    embedder: tuple[Callable,keras.Model],
    evaluator: tuple[Callable,keras.Model],
    x: list[str],
    y: list[str],
    labels: np.ndarray,
) -> float:
    scores = eval(embedder, evaluator, x, y)
    labels = np.asarray(labels)
    unique_scores = np.unique(scores)
    if len(unique_scores) == 1:
        return float(unique_scores[0])
    candidates = (unique_scores[:-1] + unique_scores[1:]) / 2
    best_threshold = candidates[0]
    best_accuracy = -1.0
    for t in candidates:
        accuracy = ((scores >= t).astype(int) == labels).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = t
    return float(best_threshold)

def classification_metrics(
    embedder: tuple[Callable,keras.Model],
    evaluator: tuple[Callable,keras.Model],
    x: list[str],
    y: list[str],
    labels: np.ndarray,
    threshold: float,
) -> dict:
    scores = eval(embedder, evaluator, x, y)
    labels = np.asarray(labels)
    preds = (scores >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    total = tp + fp + tn + fn
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / total if total > 0 else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }