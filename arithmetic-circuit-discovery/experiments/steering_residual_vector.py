import torch
from transformer_lens import HookedTransformer

def test_full_dense_patch(model_name="microsoft/Phi-3-mini-4k-instruct", layer=14):
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(model_name, device=device)

    clean_prompt = "Calculate:\n12 + 7 = 19\n14 + 23 = "
    corrupt_prompt = "Calculate:\n12 + 7 = 19\n11 + 14 = "

    clean_tokens = model.to_tokens(clean_prompt)
    corrupt_tokens = model.to_tokens(corrupt_prompt)

    with torch.no_grad():
        initial_clean_logits = model(clean_tokens)

    ans_tok = initial_clean_logits[0, -1].argmax(dim=-1).item()
    target_str = model.tokenizer.decode([ans_tok])
    print(f"\nTarget token ID: {ans_tok} (String: '{target_str}')")

    # 1. Run clean and cache the ENTIRE residual stream at Layer 14
    hook_name = f"blocks.{layer}.hook_resid_post"
    with torch.no_grad():
        _, clean_cache = model.run_with_cache(clean_tokens, names_filter=hook_name)
    clean_resid = clean_cache[hook_name]

    # --- EXPERIMENT: FULL DENSE PATCHING ---
    def full_patch_hook(resid, hook):
        # Overwrite the ENTIRE 3,072-dim state at the last token position
        resid[:, -1, :] = clean_resid[:, -1, :]
        return resid

    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hook_name, full_patch_hook)]):
            patched_logits = model(corrupt_tokens)

        patched_prob = patched_logits[0, -1].softmax(dim=-1)[ans_tok].item()

    # Run Baselines
    with torch.no_grad():
        clean_prob = initial_clean_logits[0, -1].softmax(dim=-1)[ans_tok].item()
        corrupt_prob = model(corrupt_tokens)[0, -1].softmax(dim=-1)[ans_tok].item()

    print(f"\n--- Full Dense Activation Patching (L{layer}) ---")
    print(f"Clean Probability (predicting '{target_str}'):   {clean_prob:.4f}")
    print(f"Corrupt Probability (predicting '{target_str}'): {corrupt_prob:.4f}")
    print(f"Patched (Sufficiency):    {patched_prob:.4f} <- Did we recover the thought?")

test_full_dense_patch()