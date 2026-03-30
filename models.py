import keras
import keras_hub
import numpy as np
import tensorflow as tf
from functools import partial
from typing import Callable

def tail_preprocessor(preprocessor: keras_hub.models.DistilBertPreprocessor, sequence_length: int, x: list[str]) -> dict[str,tf.Tensor]:
    tokens = tf.ragged.constant(preprocessor.tokenizer(x)) # type: ignore
    tokens = tokens[:, -(sequence_length-2):]
    batch_size = tf.shape(tokens)[0]
    start_col = tf.fill([batch_size, 1], 101)
    end_col = tf.fill([batch_size, 1], 102)
    result = tf.concat([start_col, tokens, end_col], axis=1).to_tensor(default_value=0, shape=[None, 510])
    mask = result > 0
    return {
        "token_ids": result,
        "padding_mask": mask,
    }

def masked_sum(args):
    embeddings, mask = args
    mask = keras.ops.expand_dims(keras.ops.cast(mask, embeddings.dtype), axis=-1)
    masked_embeddings = embeddings * mask
    return keras.ops.sum(masked_embeddings, axis=1)

def create_model(sequence_length: int) -> tuple[Callable, keras.Model]:
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
    outputs = keras.layers.Lambda(masked_sum)([x, mask])
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError()
    )
    return partial(tail_preprocessor, preprocessor, sequence_length), model

def predict(
    model: tuple[Callable,keras.Model],
    x: list[str],
) -> np.ndarray:
    p, m = model
    return m.predict(p(x), batch_size=16)

def fit(
    model: tuple[Callable,keras.Model],
    x: list[str],
    y: np.ndarray,
    epochs: int = 20,
    batch_size: int = 16
) -> tuple[Callable,keras.Model]:
    p, m = model
    preprocessed = p(x)
    m.fit(preprocessed, y, epochs=epochs, batch_size=batch_size)
    return p, m

def eval(
    embedder: tuple[Callable,keras.Model],
    evaluator: tuple[Callable,keras.Model],
    x: list[str],
    y: list[str],
) -> np.ndarray:
    x_embed = predict(embedder, x)
    y_embed = predict(evaluator, y)
    return np.linalg.norm(x_embed - y_embed, axis=1)