import json
import modal

MODEL = "bcywinski/gemma-2-9b-it-user-male"
PROMPTS_PATH = "prompts_gendersexbathroom.json"
OUT_PATH = "response.json"
TRIES = 10

app = modal.App("gender-secret-test")
volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch", "transformer_lens", "transformers", "huggingface_hub", "tqdm", "peft",
).pip_install("transformers>=4.46")

@app.function(
        image=image,
        gpu="A100-80GB",
        volumes={"/root/.cache/huggingface": volume},
        timeout=1800,
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
    for n in range(TRIES):
        for i, row in enumerate(tqdm(prompts)):
            prompt = row["prompt"]
            message = [{"role": "user", "content": prompt}]
            text = model.tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )

            response = model.generate(text, max_new_tokens=100, verbose=False, do_sample=True, temperature=0.8)
            responses.append({"model": "gender-secret", "prompt": prompt, "response": response})
    del model, hf_model, peft_model, base_model
    torch.cuda.empty_cache()

    base_model_clean = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch.bfloat16)
    base_model_tl = HookedTransformer.from_pretrained(base_model_name, hf_model=base_model_clean, dtype=torch.bfloat16)
    for n in range(TRIES):
            for i, row in enumerate(tqdm(prompts)):
                prompt = row["prompt"]
                message = [{"role": "user", "content": prompt}]
                text = base_model_tl.tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
    
                response = base_model_tl.generate(text, max_new_tokens=100, verbose=False, do_sample=True, temperature=0.8)
                responses.append({"model": "base", "prompt": prompt, "response": response})
    return responses

@app.local_entrypoint()
def main():
    with open(PROMPTS_PATH) as f:
        rows = json.load(f)

    responses = test_model_on_sex.remote(MODEL, "google/gemma-2-9b-it", rows)
    with open(OUT_PATH, "w") as f:
            json.dump(responses, f, indent=2)
    print(f"Saved {len(responses)} responses to {OUT_PATH}")

