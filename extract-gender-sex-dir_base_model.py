import json
import modal
import torch

MODEL = "bcywinski/gemma-2-9b-it-user-male"
PROMPTS_PATH = "prompts.json"
OUT_PATH = "response.pt"

app = modal.App("gender-secret-test")
volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch", "transformer_lens", "transformers", "huggingface_hub", "tqdm"
).pip_install("transformers>=4.46")

@app.function(
        image=image,
        gpu="A100-80GB",
        volumes={"/root/.cache/huggingface": volume},
        timeout=1800,
        secrets=[modal.Secret.from_name("huggingface-secret")],
)
def extract_gender_sex_dir(hf_model_name: str, base_model_name: str, prompts: list[dict]):
    import re
    import torch
    from tqdm import tqdm
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch.bfloat16)
    model = HookedTransformer.from_pretrained(base_model_name, hf_model=base_model, dtype=torch.bfloat16)

    responses = []
    for i, row in enumerate(tqdm(prompts)):
        prompt = row["prompt"]
        extract_at_last_token_for = row["extract"]

        message = [{"role": "user", "content": prompt}]
        text = model.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )

        pattern = re.compile(r'\b' + re.escape(extract_at_last_token_for.lower()) + r'\b')
        match = pattern.search(text.lower())
        if match is None:
            raise ValueError(f"Could not locate word '{extract_at_last_token_for}' in prompt: {prompt}")
        extract_at_index_last = match.end()
        encoding = model.tokenizer(text, return_offsets_mapping=True)
        offsets = encoding["offset_mapping"]

        try:
            target_pos = next(
                 i for i, (s, e) in enumerate(offsets)
                    if s < extract_at_index_last and e >= extract_at_index_last
            )
        except StopIteration:
             raise ValueError(f"Could not locate token span for prompt: {prompt}")

        input = torch.tensor(encoding["input_ids"]).unsqueeze(0).to(model.cfg.device)

        decoded = model.tokenizer.decode([input[0, target_pos].item()])

        layer_vectors = {}
        _, cache = model.run_with_cache(input)
        for layer in range(model.cfg.n_layers):
            resid = cache["resid_post", layer]
            layer_vectors[layer] = resid[0, target_pos, :].cpu().float()
        responses.append({
            "prompt": prompt,
            "extract": extract_at_last_token_for,
            "decoded_token_at_extraction_point": decoded,
            "activations": layer_vectors,
            "category": row["category"]
        })

    del model, base_model
    torch.cuda.empty_cache()

    return responses

@app.local_entrypoint()
def main():
    with open(PROMPTS_PATH) as f:
        rows = json.load(f)

    responses = extract_gender_sex_dir.remote(MODEL, "google/gemma-2-9b-it", rows)
    torch.save(responses, OUT_PATH)
    print(f"Saved responses to {OUT_PATH}")

