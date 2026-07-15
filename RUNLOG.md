
## Run 0: Baseline
* **Hypothesis:** Just getting a feel for where we stand with the out-of-the-box starter code.
* **What Changed:** Absolutely nothing. Just ran the unmodified `train.py` straight up. 
* **Dev BPB:** 2.3718 (Params: 1,339,840)
* **Conclusion:** Okay, baseline is set. The score is pretty rough, which makes sense since it's missing basic stuff like a proper learning rate schedule. Also, the basic byte-level tokenizer is going to be problematic on the Hindi text, so that needs an upgrade.
## Run 1: Optimizer Upgrade
* **Hypothesis:** AdamW + Cosine LR schedule + Grad clip will stabilize and speed up learning.
* **What Changed:** Swapped baseline optimizer in train.py for AdamW (lr=6e-4) with warmup and cosine decay.
* **Dev BPB:** 2.3554 (Params: 1,339,840)
* **Conclusion:** Success. Loss dropped.
## Run 2: Weight Tying
* **Hypothesis:** Tying the input embedding and output projection weights will reduce the total parameter count, freeing up budget for future upscaling.
* **What Changed:** Changed `tie_weights = False` to `True` in `model.py`.
* **Dev BPB:** 2.3982 (Params: 1,298,880)
* **Conclusion:** Parameter count successfully dropped by roughly 40k. The BPB regressed slightly because the model's overall capacity decreased. We now have safe headroom under the 2M cap to increase the embedding dimension or add layers.
## Run 3: BPE Tokenizer
* **Hypothesis:** A byte-level tokenizer was wasting context on multi-byte Hindi characters. BPE will compress the sequence length and improve learning efficiency.
* **What Changed:** Replaced byte tokenizer with a custom BPE tokenizer (vocab 512) in `tokenizer.py`.
* **Dev BPB:** 2.3001 (Params: 1,339,840)
* **Conclusion:** Massive success. The model is learning the mixed English/Hindi data much faster. The parameter count rose slightly due to the larger vocab size but under budget.
## Run 4: SwiGLU Activation
* **Hypothesis:** SwiGLU provides a richer representation space than standard GELU MLPs, increasing learning capacity for roughly the same parameter count.
* **What Changed:** Replaced standard MLP block with a custom SwiGLU module.
* **Dev BPB:** 2.2946 (Params: 1,335,360)
* **Conclusion:** Success. BPB improved further and we still have plenty of parameter budget.
## Run 5: Parameter Upscale
* **Hypothesis:** Utilizing the freed-up parameter budget to increase the model's depth/width will increase its raw capacity.
* **What Changed:** Increased `n_embd` and/or `n_layer` in `model.py` to push closer to the 2M cap.
* **Dev BPB:** 2.2970 (Params: 1,643,520)
* **Conclusion:** Safely increased model capacity without blowing past the hard cap. The score remained stable, indicating we might need to adjust regularization for this new size.
