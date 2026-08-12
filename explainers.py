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


def similarity_to_html(
    token_sims: list[dict],
    title: str = "token similarity",
    min_sim: float = -1.0,
    max_sim: float = 1.0,
) -> str:
    if max_sim <= min_sim:
        raise ValueError("max_sim must be greater than min_sim")

    def _color(sim: float) -> str:
        t = (sim - min_sim) / (max_sim - min_sim)
        t = max(0.0, min(1.0, t))
        red = int(round((1.0 - t) * 255))
        green = int(round(t * 255))
        return f"rgb({red},{green},0)"

    escaped_title = (title or "").replace("<", "&lt;").replace(">", "&gt;")
    rows = []
    for entry in token_sims:
        token = str(entry["token"])
        sim = float(entry["sim"])
        token_html = token.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        token_html = token_html.replace("\n", "<br>")
        color = _color(sim)
        rows.append(
            f'<span title="{sim:.4f}" '
            f'style="background-color:{color};color:#000;'
            f'padding:1px 2px;border-radius:3px;'
            f'white-space:pre-wrap;">{token_html}</span>'
        )

    body = " ".join(rows)
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{escaped_title}</title>\n"
        "<style>\n"
        "body{font-family:monospace;font-size:16px;line-height:1.6;}\n"
        "h1{font-size:18px;}\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>{escaped_title}</h1>\n"
        f"<p>{body}</p>\n"
        "</body>\n"
        "</html>\n"
    )