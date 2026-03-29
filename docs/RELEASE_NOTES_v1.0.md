## [v1.0] — Florence-2 for ComfyUI (transformers 5.x compatible)

**Tag:** `v1.0`  
**Focus:** Reliable inference with **transformers ≥ 5.0** (validated on **5.4.0**), while keeping **transformers 4.x** usable where the version-gated paths apply.

### Summary

This release restores end-to-end Florence-2 behavior in ComfyUI after upgrading the Hugging Face `transformers` stack to v5.x. Earlier “fixes” often removed crashes but left **empty captions**, **broken prompts**, or **dtype / shape errors**; this fork addresses the underlying causes: API drift in `transformers`, image preprocessor semantics, and a latent bug in `generate()` that only became visible under v5.

---

### Highlights (user-visible)

* **transformers 5.x:** Model load and attention backend selection work without import errors or “does not support Flash Attention 2” failures.
* **Correct outputs again:** Captioning, tagging, OD/OCR-style tasks, and DocVQA paths produce meaningful text instead of silence or garbage.
* **Stable image pipeline:** Images are resized and normalized as Florence-2 expects (768×768 square path), avoiding `uint8` vs half weights mismatches and “square feature maps” assertions.
* **Safer default decoding:** `num_beams` default is **1** (greedy); raise it if you need beam search after confirming behavior on your GPU.

---

### Technical themes (map to detailed docs)

| Theme | What changed (short) | Deep dive |
| --- | --- | --- |
| **A. transformers 5.x + Flash Attention 2** | Weight tying without removed helpers, conditional post_init, FA import shim, _supports_flash_attn flags, forced_bos_token_id ordering, generation_config sync after custom load | [FIX_01_transformers5_api_compatibility.md](https://github.com/ussoewwin/ComfyUI-Florence2/blob/v1.0/docs/FIX_01_transformers5_api_compatibility.md) |
| **B. Image preprocessing** | Do not forward None into CLIPImageProcessor for resize/normalize/rescale so v5 uses processor defaults | [FIX_02_image_preprocessing.md](https://github.com/ussoewwin/ComfyUI-Florence2/blob/v1.0/docs/FIX_02_image_preprocessing.md) |
| **C. Generation** | Pass merged attention_mask into language_model.generate(); fallback mask from inputs_embeds when needed | [FIX_03_attention_mask_generation.md](https://github.com/ussoewwin/ComfyUI-Florence2/blob/v1.0/docs/FIX_03_attention_mask_generation.md) |

---

### Files touched (high level)

* `modeling_florence2.py` — tying, init guards, FA utilities, `generate()` attention mask forwarding, support flags.
* `configuration_florence2.py` — `forced_bos_token_id` handling after `super().__init__()`.
* `processing_florence2.py` — conditional kwargs to the image processor.
* `nodes.py` — `load_model()` for transformers 5.x path (tokenizer resize, `generation_config`), `num_beams` default.

---

### Compatibility

* **Tested:** `transformers` **5.4.0**, PyTorch 2.x (CUDA), Florence-2 checkpoints (e.g. large / large-ft).
* **Backward compatibility:** Version checks and `hasattr` shims aim to keep **transformers 4.x** workflows working on code paths that still use the older loading style.

---

### Upgrade notes

* If you previously relied on **beam search** (`num_beams` > 1), re-test workflows; the **default is now 1** to reduce failure modes and cost. Increase intentionally if needed.
* For PRs or forks upstream, cite **FIX_01–03** (under `docs/`) so reviewers can follow rationale without duplicating long explanations in the PR body.
