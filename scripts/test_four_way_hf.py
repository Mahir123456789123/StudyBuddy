import sys
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.abspath("."))
from app.config import settings
from app.ingestion import get_embedding_function, load_vectorstore
from app.retriever import CourseRetriever

print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(settings.HF_MODEL_NAME_OR_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    settings.HF_MODEL_NAME_OR_PATH,
    torch_dtype=torch.float16,
    device_map="cuda"
)
peft_model = PeftModel.from_pretrained(base_model, settings.LORA_ADAPTER_PATH)
peft_model.eval()

print("Loading retriever...")
emb_fn = get_embedding_function(settings.EMBEDDING_MODEL_NAME)
vectorstore = load_vectorstore(settings.INDEX_PATH, emb_fn)
retriever = CourseRetriever(vectorstore, k=settings.RETRIEVER_K)

test_q = "What mathematical formula represents the weight update in LoRA?"
docs = retriever.retrieve(test_q)
context_str, sources = retriever.format_context_and_sources(docs)

def generate_text(model, prompt, use_adapter=True, max_tokens=60):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        if use_adapter:
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        else:
            with model.disable_adapter():
                outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

print("\n1. Base Model (No adapter, No RAG):")
resp_base = generate_text(peft_model, f"Question: {test_q}\nAnswer:", use_adapter=False)
print(resp_base)

print("\n2. Base + RAG (No adapter, With RAG):")
resp_base_rag = generate_text(peft_model, f"Context:\n{context_str}\n\nQuestion: {test_q}\nAnswer:", use_adapter=False)
print(resp_base_rag)

print("\n3. Fine-Tuned Only (With adapter, No RAG):")
resp_ft = generate_text(peft_model, f"<|im_start|>system\nYou are StudyBuddy, an expert course tutor.<|im_end|>\n<|im_start|>user\n{test_q}<|im_end|>\n<|im_start|>assistant\n", use_adapter=True)
print(resp_ft)

print("\n4. Fine-Tuned + RAG (With adapter, With RAG):")
prompt_ft_rag = f"<|im_start|>system\nYou are StudyBuddy, an expert course tutor.<|im_end|>\n<|im_start|>user\nContext:\n{context_str}\n\nQuestion: {test_q}<|im_end|>\n<|im_start|>assistant\n"
resp_ft_rag = generate_text(peft_model, prompt_ft_rag, use_adapter=True)
print(resp_ft_rag)
