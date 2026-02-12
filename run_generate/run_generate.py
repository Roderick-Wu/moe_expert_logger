# run_generate.py
import os, json, time, random
import pyarrow.parquet as pq

#os.environ["VLLM_MOE_PADDING"] = "0"
#os.environ["VLLM_USE_V1"] = "0" 
#os.environ["VLLM_TARGET_DEVICE"] = "cpu"
#os.environ["VLLM_CPU_KVCACHE_SPACE"] = "4" 
#os.environ["VLLM_CPU_MOE_PREPACK"] = "1" 

from vllm import LLM, SamplingParams

random.seed(1234)

def inference():
    table = pq.read_table("../datasets/gsm8k_main/main/test-00000-of-00001.parquet")
    prompts = table.column("question").to_pylist()[:25]

    print("Questions =====================")
    for i, prompt in enumerate(prompts):
        print(f"{i}: {prompt}")
    print("===============================")

    sp = SamplingParams(temperature=0.0, max_tokens=128)
    llm = LLM(
        model="../models/Qwen1.5-MoE-A2.7B-Chat",
        load_format="safetensors",
        gpu_memory_utilization=0.8,
        max_model_len=512,
        enforce_eager=True, # having issues logging because of the cuda graph it seems
    )

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    t1 = time.time()
    # Required timing artifact:
    json.dump({"no_log": {"wall_time_sec": t1 - t0,
                          "tokens_generated": sum(len(o.outputs[0].token_ids) for o in outputs)}},
          open("timing.json","w"))

if __name__ == "__main__":
    inference()


