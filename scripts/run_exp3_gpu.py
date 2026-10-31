import os
import time
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType

assert torch.cuda.is_available(), "CUDA must be available!"
device = 'cuda'
print(f"Running on GPU: {torch.cuda.get_device_name(0)}")

model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompts = [
    'Explain the role of rank r in LoRA.',
    'What is FAISS IndexFlatL2?',
    'Define NormalFloat4 (NF4) quantization.'
]

def profile_configuration(config_name, quant_config=None, dtype=torch.float16):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=dtype if quant_config is None else None,
        device_map='auto'
    )
    
    alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    res_mb = torch.cuda.memory_reserved() / (1024 ** 2)
    
    latencies = []
    gen_tokens = []
    for p in prompts:
        inp = tokenizer(f'User: {p}\nAssistant:', return_tensors='pt').to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=40)
        latencies.append((time.perf_counter() - t0) * 1000)
        gen_tokens.append(out.shape[1] - inp.input_ids.shape[1])
    
    del model
    torch.cuda.empty_cache()
    return {
        'Config': config_name,
        'Allocated VRAM (MB)': round(alloc_mb, 1),
        'Reserved VRAM (MB)': round(res_mb, 1),
        'Avg Latency (ms)': round(sum(latencies) / len(latencies), 1),
        'Avg Generated Tokens': round(sum(gen_tokens) / len(gen_tokens), 1)
    }

results = []
print("1. Profiling Baseline FP16...")
results.append(profile_configuration('Baseline FP16', dtype=torch.float16))

print("2. Profiling INT8 Quantization...")
bnb_int8 = BitsAndBytesConfig(load_in_8bit=True)
results.append(profile_configuration('INT8 Quantization', quant_config=bnb_int8))

print("3. Profiling INT4 / NF4 Double Quant...")
bnb_nf4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
results.append(profile_configuration('INT4 / NF4 Double Quant', quant_config=bnb_nf4))

df_quant = pd.DataFrame(results)
print("\n" + df_quant.to_markdown(index=False))

results_dir = 'eval/results'
os.makedirs(results_dir, exist_ok=True)
df_quant.to_csv(os.path.join(results_dir, 'exp3_quantization_results.csv'), index=False)
with open(os.path.join(results_dir, 'exp3_quantization_results.md'), 'w') as f:
    f.write('# Experiment 3: Quantization & QLoRA Profiling Results\n\n' + df_quant.to_markdown(index=False) + '\n')
print(f"Exp 3 results saved to {results_dir}")
