import models
import json

def main():
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
    a_test = aligned[split_index:]
    m_test = misaligned[split_index:]
    print("train", len(a_train))
    print("test", len(a_test))
    embedder = models.create_model()
    evaluator = models.create_model()
    y = models.predict(embedder, p_train)
    evaluator = models.fit(evaluator, a_train, y, epochs=100)
    test_aligned = models.eval(embedder, evaluator, p_test, a_test)
    print(test_aligned)
    test_misaligned = models.eval(embedder, evaluator, p_test, m_test)
    print(test_misaligned)

if __name__ == "__main__":
    main()