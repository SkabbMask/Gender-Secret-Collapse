import json
import modal

MODEL = "bcywinski/gemma-2-9b-it-user-male"
PROMPTS_PATH = "prompts.json"
OUT_PATH = "response.json"

app = modal.App("gender-secret-test")
volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch", "transformer_lens", "transformers", "huggingface_hub", "tqdm", "peft",
).pip_install("transformers>=4.46")

@app.function(
        image=image,
        gpu="A100-80GB",
        volumes={"/root/.cache/huggingface": volume},
        timeout=600,
        secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_model_on_sex(hf_model_name: str, base_model_name: str, prompts: list[dict]):
    import torch
    from tqdm import tqdm
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch.bfloat16)
    peft_model = PeftModel.from_pretrained(base_model, hf_model_name)
    hf_model = peft_model.merge_and_unload()
    model = HookedTransformer.from_pretrained(base_model_name, hf_model=hf_model, dtype=torch.bfloat16)

    responses = []
    for i, row in enumerate(tqdm(prompts)):
        prompt = row["prompt"]
        message = [{"role": "user", "content": prompt}]
        text = model.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )

        response = model.generate(text, max_new_tokens=100, verbose=False)
        responses.append({"prompt": prompt, "response": response})
    return responses

@app.local_entrypoint()
def main():
    with open(PROMPTS_PATH) as f:
        rows = json.load(f)

    responses = test_model_on_sex.remote(MODEL, "google/gemma-2-9b-it", rows)
    with open(OUT_PATH, "w") as f:
            json.dump(responses, f, indent=2)
    print(f"Saved {len(responses)} responses to {OUT_PATH}")

