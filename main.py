import os
import models
import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import explainers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

def main():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    train_split = 0.8
    with open("data/prompts.json") as file:
        prompts = json.load(file)
    with open("data/aligned.json") as file:
        aligned = json.load(file)
    with open("data/misaligned.json") as file:
        misaligned = json.load(file)
    log.info(f"prompts {len(prompts)}")
    log.info(f"aligned {len(aligned)}")
    log.info(f"misaligned {len(misaligned)}")
    if len(prompts) != len(aligned) or len(aligned) != len(misaligned):
        raise RuntimeError("Lengths do not match!")
    split_index = int(len(prompts) * train_split)
    p_train = prompts[:split_index]
    p_test = prompts[split_index:]
    a_train = aligned[:split_index]
    m_train = misaligned[:split_index]
    a_test = aligned[split_index:]
    m_test = misaligned[split_index:]
    log.info(f"train {len(a_train)}")
    log.info(f"test {len(a_test)}")

    embedder = models.create_model(sequence_length=128, loss="cosine_similarity")
    evaluator = models.create_model(sequence_length=128, loss="cosine_similarity")

    log.info("pretraining aligned similarity")
    pre_align = models.eval(embedder, evaluator, p_test, a_test)
    log.info(f"mean: {pre_align.mean()}")
    log.info("pretraining misaligned similarity")
    pre_misalign = models.eval(embedder, evaluator, p_test, m_test)
    log.info(f"mean: {pre_misalign.mean()}")

    log.info("pretraining evaluator embedding variance (test set)")
    pre_var = models.get_variances(evaluator, a_test)
    log.info(f"mean pairwise distance: {pre_var.mean()}")

    log.info("pre-computing prompt embeddings (frozen)...")
    p_embeds = models.predict(embedder, p_train)
    log.info(f"done, shape: {p_embeds.shape}")

    models.train_evaluator_contrastive(
        evaluator, p_embeds, p_train, a_train, m_train,
        epochs=30, batch_size=4, temperature=0.1,
    )

    log.info("posttraining aligned similarity (higher = better)")
    post_align = models.eval(embedder, evaluator, p_test, a_test)
    log.info(f"mean: {post_align.mean()}")
    log.info("posttraining misaligned similarity")
    post_misalign = models.eval(embedder, evaluator, p_test, m_test)
    log.info(f"mean: {post_misalign.mean()}")

    log.info("posttraining evaluator embedding variance (test set)")
    post_var = models.get_variances(evaluator, a_test)
    log.info(f"mean pairwise distance: {post_var.mean()}")

    log.info("aligned vs misaligned separation:")
    log.info(f"aligned   - mean sim: {post_align.mean()} std: {post_align.std()}")
    log.info(f"misaligned - mean sim: {post_misalign.mean()} std: {post_misalign.std()}")

    log.info("determining best classification threshold (train set)")
    threshold = models.find_best_threshold(
        embedder, evaluator,
        p_train + p_train, a_train + m_train,
        np.array([1] * len(a_train) + [0] * len(m_train)),
    )
    log.info(f"best threshold: {threshold}")

    log.info("classification metrics (test set)")
    metrics = models.classification_metrics(
        embedder, evaluator,
        p_test + p_test, a_test + m_test,
        np.array([1] * len(a_test) + [0] * len(m_test)),
        threshold,
    )
    log.info(json.dumps(metrics, indent=2))
    with open("results/classification_metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    log.info("generating per-sample explanations")
    for kind, responses, overall_sims in (
        ("aligned", a_test, post_align),
        ("misaligned", m_test, post_misalign),
    ):
        raw_dir = os.path.join("results", "explanations", kind, "raw")
        html_dir = os.path.join("results", "explanations", kind, "html")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(html_dir, exist_ok=True)
        for i, response in enumerate(responses):
            ref = models.predict(embedder, p_test[i:i + 1])
            token_sims = explainers.token_similarity_to_reference(
                evaluator, response, ref
            )
            with open(os.path.join(raw_dir, f"{i}.json"), "w") as file:
                json.dump(token_sims, file, indent=2)
            html = explainers.similarity_to_html(
                token_sims,
                title=f"{kind} #{i} (prompt: {p_test[i]!r})",
                overall_sim=float(overall_sims[i]),
                threshold=float(threshold),
            )
            with open(os.path.join(html_dir, f"{i}.html"), "w") as file:
                file.write(html)
    log.info("explanations written under results/explanations/")

    plt.figure()
    plt.hist(post_align, bins=20, alpha=0.6, label="aligned")
    plt.hist(post_misalign, bins=20, alpha=0.6, label="misaligned")
    plt.axvline(threshold, color="red", linestyle="--", label=f"threshold = {threshold:.3f}")
    plt.title(f"test accuracy: {metrics['accuracy']:.3f}")
    plt.xlabel("cosine similarity")
    plt.ylabel("frequency")
    plt.legend()
    plt.savefig("results/post_alignment_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hist(post_align - post_misalign, bins=30)
    plt.xlabel("aligned sim - misaligned sim")
    plt.ylabel("frequency")
    plt.savefig("results/post_separation_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()