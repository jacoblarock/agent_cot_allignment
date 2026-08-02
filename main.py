import models
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import explainers

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
    print("prompts", len(prompts))
    print("aligned", len(aligned))
    print("misaligned", len(misaligned))
    if len(prompts) != len(aligned) or len(aligned) != len(misaligned):
        raise RuntimeError("Lengths do not match!")
    split_index = int(len(prompts) * train_split)
    p_train = prompts[:split_index]
    p_test = prompts[split_index:]
    a_train = aligned[:split_index]
    m_train = misaligned[:split_index]
    a_test = aligned[split_index:]
    m_test = misaligned[split_index:]
    print("train", len(a_train))
    print("test", len(a_test))

    embedder = models.create_model(sequence_length=128, loss="cosine_similarity")
    evaluator = models.create_model(sequence_length=128, loss="cosine_similarity")

    print("pretraining aligned similarity")
    pre_align = models.eval(embedder, evaluator, p_test, a_test)
    print("mean:", pre_align.mean())
    print("pretraining misaligned similarity")
    pre_misalign = models.eval(embedder, evaluator, p_test, m_test)
    print("mean:", pre_misalign.mean())

    print("\npretraining evaluator embedding variance (test set)")
    pre_var = models.get_variances(evaluator, a_test)
    print("mean pairwise distance:", pre_var.mean())

    print("\npre-computing prompt embeddings (frozen)...")
    p_embeds = models.predict(embedder, p_train)
    print("done, shape:", p_embeds.shape)

    models.train_evaluator_contrastive(
        evaluator, p_embeds, p_train, a_train, m_train,
        epochs=30, batch_size=4, temperature=0.1,
    )

    print("\nposttraining aligned similarity (higher = better)")
    post_align = models.eval(embedder, evaluator, p_test, a_test)
    print("mean:", post_align.mean())
    print("posttraining misaligned similarity")
    post_misalign = models.eval(embedder, evaluator, p_test, m_test)
    print("mean:", post_misalign.mean())

    print("\nposttraining evaluator embedding variance (test set)")
    post_var = models.get_variances(evaluator, a_test)
    print("mean pairwise distance:", post_var.mean())

    print("\naligned vs misaligned separation:")
    print("aligned   - mean sim:", post_align.mean(), "std:", post_align.std())
    print("misaligned - mean sim:", post_misalign.mean(), "std:", post_misalign.std())

    print("\ndetermining best classification threshold (train set)")
    threshold = models.find_best_threshold(
        embedder, evaluator,
        p_train + p_train, a_train + m_train,
        np.array([1] * len(a_train) + [0] * len(m_train)),
    )
    print("best threshold:", threshold)

    print("\nclassification metrics (test set)")
    metrics = models.classification_metrics(
        embedder, evaluator,
        p_test + p_test, a_test + m_test,
        np.array([1] * len(a_test) + [0] * len(m_test)),
        threshold,
    )
    print(json.dumps(metrics, indent=2))
    with open("data/classification_metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    ref = models.predict(
        embedder,
        p_test[0:1]
    )
    test_exp = {
        "aligned": explainers.token_similarity_to_reference(
            evaluator,
            a_test[0],
            ref
        ),
        "misaligned": explainers.token_similarity_to_reference(
            evaluator,
            m_test[0],
            ref
        ),
    }

    with open("data/test_exp.json", "w") as file:
        json.dump(test_exp, file, indent=2)

    plt.figure()
    plt.hist(post_align, bins=20, alpha=0.6, label="aligned")
    plt.hist(post_misalign, bins=20, alpha=0.6, label="misaligned")
    plt.axvline(threshold, color="red", linestyle="--", label=f"threshold = {threshold:.3f}")
    plt.title(f"test accuracy: {metrics['accuracy']:.3f}")
    plt.xlabel("cosine similarity")
    plt.ylabel("frequency")
    plt.legend()
    plt.savefig("data/post_alignment_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hist(post_align - post_misalign, bins=30)
    plt.xlabel("aligned sim - misaligned sim")
    plt.ylabel("frequency")
    plt.savefig("data/post_separation_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()