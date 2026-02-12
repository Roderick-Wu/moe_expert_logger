# MoE Expert Logger and Histogram in vLLM
For Cerebras Assessment

In MoE path, add flag-gated logger that, per token, records routing information. Logs only one MoE layer.
Modifies vLLM so we can log which experts are selected per token, for select layers, plots expert-usage histogram. 

When flags are not set, behavior is unchanged. 

## vLLM Modifications

We create a a logger class at `./vllm/vllm/model_executor/layers/fused_moe/moe_expert_log.py`. 

To determine where to look, we can identify where things are located by poking around `fused_moe` directory. 

FusedMoE (class used for all MoE forwards?) exists at `./vllm/vllm/model_executor/layers/fused_moe/layer.py`. The FusedMoE class always creates router using create_fused_moe_router from `./vllm/vllm/model_executor/layers/fused_moe/router_factory.py`. All routers built by this function inherit from template class BaseRouter, found in `./vllm/vllm/model_executor/layers/fused_moe/base_router.py`. We hook this class's select_experts function after it computes topk_weights and topk_ids. 

```
        # Step 1: Validate EPLB state
        self._validate_eplb_state()

        # Step 2: Get indices type.
        indices_type = self._get_indices_type()

        # Step 3: Compute routing (delegated to subclass)
        topk_weights, topk_ids = self._compute_routing(
            hidden_states, router_logits, indices_type
        )

        moe_logger.log(topk_ids, topk_weights) ### HOOKED HERE ###

        # Capture logical ids before EPLB mapping.
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)

        # Step 4: Apply EPLB mapping
        topk_ids = self._apply_eplb_mapping(topk_ids)

        # Step 5: Convert indices dtype
        topk_ids = self._convert_indices_dtype(topk_ids, indices_type)

        return topk_weights, topk_ids
```


### Usage
```
export VLLM_LOG_MOE="vllm_moe_log.jsonl"    # Path to log file
export VLLM_LOG_LAYER="0"                   # Layer to log, this example logs first layer
export VLLM_NUM_MOE_LAYERS="24"             # Total number of layers, need this for a counter in the logger or else there's more to modify
python run_generate.py                      # Simple script for inference
```


### Notes

I had to run this on compute-canada cluster eventually. Originally I was going to try running this locally. So I found a quantized version of the model at
https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4
. 
However, even quantized my laptop gpu only has 6GB of vram. I tried hybrid approach combining gpu memory and cpu memory but ran into issues with the vllm engine for quantized moe models. I then tried to run inference with only cpu (16GB RAM on my laptop), but ran into many issues. vLLM was missing quite a few cpu compute kernels for quantized models -- specifically gptq (for this quantization). Needed to add rudimentary torch implementations for gemm, rope, and router. Not sure if it was working, but it was never able to complete inference (wsl repeated crashing). But during crashes the stack trace was was useful for identifying where operations occured. This let me find where the router was easily. 

On compute-canada, just needed to create a simple venv, load basic modules, install editable vLLM locally, and run inference through slurm. This cluster A100s. 

```
module load python cuda gcc arrow
cd vllm
pip install -e .
```

### AI Usage

I used AI for reading most of the long vLLM error outputs. It was useful for quickly identifying where errors were occuring (especially when I was trying to figure it out on CPU). For coding, I used it to generate the script to plot the .jsonl output (`./run_generate/plot_info.py`). The script is small and very quick to run, verify, and debug. 

### Results

Generally very strange expert distributions; seems that all layers (at least the ones I tried) had a very skewed distribution. Most tokens had the same 4 top experts with same probabilities -- something with prefill tokens? If we ignore the top4 for each layer, the tokens are generally quite evenly distributed among the remaining experts.

![Layer 0](/run_generate/expert_hist_layer0.png)
![Layer 12](/run_generate/expert_hist_layer12.png)
![Layer 23](/run_generate/expert_hist_layer23.png)

