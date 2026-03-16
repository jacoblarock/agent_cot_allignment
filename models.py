import keras
import keras_hub
import numpy as np

def create_model() -> tuple[keras_hub.models.DistilBertPreprocessor,keras.Model]:
    preprocessor = keras_hub.models.DistilBertPreprocessor.from_preset(
        "distil_bert_base_en",
        sequence_length=512,
    )
    backbone = keras_hub.models.DistilBertBackbone.from_preset(
        "distil_bert_base_en_uncased"
    )
    model = keras.Model(
        inputs=backbone.input,
        outputs=keras.layers.Lambda(lambda x : keras.ops.sum(x, axis=1))(backbone(backbone.input))
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError()
    )
    return preprocessor, model

def predict(
    model: tuple[keras_hub.models.DistilBertPreprocessor,keras.Model],
    x: list[str],
) -> np.ndarray:
    p, m = model
    return m.predict(p(x), batch_size=16)

def fit(
    model: tuple[keras_hub.models.DistilBertPreprocessor,keras.Model],
    x: list[str],
    y: np.ndarray,
    epochs: int = 20,
) -> tuple[keras_hub.models.DistilBertPreprocessor,keras.Model]:
    p, m = model
    preprocessed = p(x)
    m.fit(preprocessed, y, epochs=epochs, batch_size=16)
    return p, m

def eval(
    embedder: tuple[keras_hub.models.DistilBertPreprocessor,keras.Model],
    evaluator: tuple[keras_hub.models.DistilBertPreprocessor,keras.Model],
    x: list[str],
    y: list[str],
) -> np.ndarray:
    x_embed = predict(embedder, x)
    y_embed = predict(evaluator, y)
    return np.linalg.norm(x_embed - y_embed, axis=1)