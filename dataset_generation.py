import ollama
import os
import json

model = "gpt-oss:20b"

use_existing = True

def respond(prompt: str) -> str:
    return ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
        }],
        think=True
    )["message"]["content"]

def get_role_description() -> str:
    with open("prompts/role_description.txt") as file:
        prompt = file.read()
    return respond(prompt)

def generate_prompt(role: str) -> str:
    with open("prompts/role_prompt.txt") as file:
        prompt = file.read().replace("DESCRIPTION", role)
    return respond(prompt)

def cot_aligned(role: str, role_prompt: str) -> str:
    with open("prompts/cot_aligned.txt") as file:
        prompt = file.read().replace("DESCRIPTION", role).replace("PROMPT", role_prompt)
    return respond(prompt)

def cot_misaligned(role: str, role_prompt: str) -> str:
    with open("prompts/cot_misaligned.txt") as file:
        prompt = file.read().replace("DESCRIPTION", role).replace("PROMPT", role_prompt)
    return respond(prompt)

def main():
    if not os.path.isdir("data"):
        os.mkdir("data")
    roles = [get_role_description() for _ in range(10)]
    if use_existing:
        if os.path.isfile("data/prompts.json"):
            with open("data/prompts.json") as file:
                prompts = json.load(file)
        else:
            prompts = []
        if os.path.isfile("data/aligned.json"):
            with open("data/aligned.json") as file:
                aligned = json.load(file)
        else:
            aligned = []
        if os.path.isfile("data/misaligned.json"):
            with open("data/misaligned.json") as file:
                misaligned = json.load(file)
        else:
            misaligned = []
    for role in roles:
        print(role)
        for i in range(5):
            print("response", i)
            prompts.append(generate_prompt(role))
            aligned.append(cot_aligned(role, prompts[-1]))
            misaligned.append(cot_misaligned(role, prompts[-1]))
    with open("data/prompts.json", "w") as file:
        json.dump(prompts, file)
    with open("data/aligned.json", "w") as file:
        json.dump(aligned, file)
    with open("data/misaligned.json", "w") as file:
        json.dump(misaligned, file)

if __name__ == "__main__":
    main()