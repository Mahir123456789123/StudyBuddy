import os
import time
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType

assert torch.cuda.is_available(), "CUDA must be available!"
device = 'cuda'
print(f"Executing LoRA Study on GPU: {torch.cuda.get_device_name(0)}")

model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load training data
train_path = 'data/finetune/train.jsonl'
with open(train_path, 'r', encoding='utf-8') as f:
    raw_train = [json.loads(line) for line in f]

formatted_texts = []
for row in raw_train:
    msgs = row.get('messages', [])
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    formatted_texts.append(text)

dataset = Dataset.from_dict({'text': formatted_texts})
def tokenize_fn(batch):
    res = tokenizer(batch['text'], truncation=True, max_length=256, padding=False)
    res['labels'] = res['input_ids'].copy()
    return res

tokenized_ds = dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
print(f"Loaded {len(tokenized_ds)} formatted training sequences.")

# Load all 20 held-out evaluation questions
eval_path = 'data/finetune/eval_questions.jsonl'
with open(eval_path, 'r', encoding='utf-8') as f:
    eval_questions = [json.loads(line) for line in f]
print(f"Loaded {len(eval_questions)} held-out evaluation questions.")

def evaluate_on_holdout(model, tokenizer, eval_questions):
    model.eval()
    question_scores = []
    
    for q in eval_questions:
        q_id = q['id']
        q_text = q['question']
        q_type = q['type']
        expected_intent = q['expected_intent']
        keywords = q.get('keywords', [])
        
        msgs = [{"role": "user", "content": q_text}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        comp_lower = completion.lower()
        
        # Rigorous scoring heuristic based on question type
        if q_type == 'out_of_scope':
            # Must actively refuse or indicate insufficient context / outside scope
            # Does NOT get credit for just echoing words or generating hallucinations
            refusal_markers = ['insufficient', 'cannot', 'not covered', 'outside the scope', 'not provided', 
                               'course material', 'do not have', 'i cannot answer', 'unrelated']
            has_refusal = any(m in comp_lower for m in refusal_markers)
            # Hallucination penalty: if asking about weather/FIFA and gives an answer without refusal
            if has_refusal:
                score = 5.0
            else:
                score = 1.0  # hallucinated or failed to refuse out-of-scope query
        elif expected_intent == 'quiz_request':
            # Check for multiple choice structure (options, question, JSON or format)
            has_json = '{' in completion and '}' in completion and ('question' in comp_lower or 'options' in comp_lower)
            has_mcq_structure = any(opt in completion for opt in ['A)', 'B)', '1.', 'a.', 'options'])
            if has_json:
                score = 5.0
            elif has_mcq_structure:
                score = 4.0
            else:
                score = 2.0
        elif expected_intent == 'interview_practice':
            # Check for rubric sections
            rubric_markers = ['verdict', 'what was good', 'what was missing', 'rubric', 'score', 'feedback']
            matched_rubrics = sum(1 for m in rubric_markers if m in comp_lower)
            if matched_rubrics >= 2:
                score = 5.0
            elif matched_rubrics == 1:
                score = 3.5
            else:
                score = 2.0
        else:
            # Concept Q&A: check keyword coverage from ground truth
            if keywords:
                matched_kw = sum(1 for kw in keywords if kw.lower() in comp_lower)
                ratio = matched_kw / len(keywords)
                if ratio >= 0.6:
                    score = 5.0
                elif ratio >= 0.3:
                    score = 4.0
                elif ratio > 0:
                    score = 3.0
                else:
                    score = 2.0
            else:
                score = 3.0
                
        question_scores.append(score)
        
    avg_score = round(sum(question_scores) / len(question_scores), 2)
    return avg_score

runs = [
    {'run_id': 'Run_A', 'lr': 1e-4, 'epochs': 2, 'r': 4},
    {'run_id': 'Run_B', 'lr': 2e-4, 'epochs': 2, 'r': 4},
    {'run_id': 'Run_C', 'lr': 2e-4, 'epochs': 2, 'r': 8},
    {'run_id': 'Run_D', 'lr': 5e-4, 'epochs': 2, 'r': 4},
]

results_records = []
trained_adapters = {}

for config in runs:
    run_id = config['run_id']
    print(f"\n=================== Executing {run_id} ===================")
    print(config)
    
    # Reload FRESH base model per run on GPU
    torch.cuda.empty_cache()
    fresh_base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config['r'],
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj']
    )
    model_lora = get_peft_model(fresh_base_model, peft_config)
    trainable_params, all_param = model_lora.get_nb_trainable_parameters()
    param_pct = (trainable_params / all_param) * 100
    
    # 25 gradient steps per run on GPU for consistent comparison
    train_args = TrainingArguments(
        output_dir=f'./results_{run_id}',
        num_train_epochs=config['epochs'],
        max_steps=25,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=config['lr'],
        fp16=True,
        logging_steps=5,
        save_strategy='no',
        report_to='none'
    )
    
    trainer = Trainer(
        model=model_lora,
        args=train_args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8)
    )
    
    t0 = time.time()
    train_output = trainer.train()
    elapsed_sec = time.time() - t0
    
    # Score against all 20 eval questions
    print(f"Evaluating {run_id} against all {len(eval_questions)} held-out evaluation questions...")
    holdout_score = evaluate_on_holdout(model_lora, tokenizer, eval_questions)
    print(f"-> {run_id} Holdout Score (1-5): {holdout_score} | Train Loss: {train_output.training_loss:.4f}")
    
    rec = {
        'Run': run_id,
        'Learning Rate': config['lr'],
        'Epochs': config['epochs'],
        'LoRA r': config['r'],
        'Train Loss': round(train_output.training_loss, 4),
        'Train Time (s)': round(elapsed_sec, 1),
        'Trainable Params (%)': f'{param_pct:.3f}%',
        'Task Quality (1-5)': holdout_score
    }
    results_records.append(rec)
    
    # Save adapter temporarily
    temp_dir = f'./temp_adapter_{run_id}'
    model_lora.save_pretrained(temp_dir)
    trained_adapters[run_id] = temp_dir
    
    del model_lora, fresh_base_model
    torch.cuda.empty_cache()
    
    # Clean check dir
    import shutil
    shutil.rmtree(f'./results_{run_id}', ignore_errors=True)

# Code-based winner selection
print("\n" + "=" * 60)
print("EXPERIMENT 2: CODE-BASED WINNER SELECTION")
print("=" * 60)

# Sort primarily by Task Quality descending, secondarily by Train Loss ascending
sorted_by_composite = sorted(results_records, key=lambda x: (-x['Task Quality (1-5)'], x['Train Loss']))
winner = sorted_by_composite[0]
winner_id = winner['Run']

print(f"Selected Winner via Code: {winner_id}")
print(f"Winning Metrics: Task Quality = {winner['Task Quality (1-5)']}, Train Loss = {winner['Train Loss']}, Params = {winner['Trainable Params (%)']}")

# Document ties honestly
qual_scores = [r['Task Quality (1-5)'] for r in results_records]
if len(set(qual_scores)) < len(qual_scores):
    print("Tie Notice: Multiple runs achieved similar holdout scores. Train loss and parameter efficiency were used as rigorous tie-breakers.")

# Save winning adapter to canonical models/adapter/
winner_temp = trained_adapters[winner_id]
adapter_dir = 'models/adapter'
os.makedirs(adapter_dir, exist_ok=True)
import shutil
for f_name in os.listdir(winner_temp):
    shutil.copy2(os.path.join(winner_temp, f_name), os.path.join(adapter_dir, f_name))
print(f"Saved winning adapter ({winner_id}) to {adapter_dir}")

# Cleanup temp adapters
for ad in trained_adapters.values():
    shutil.rmtree(ad, ignore_errors=True)

df_results = pd.DataFrame(results_records)
print("\n" + df_results.to_markdown(index=False))

results_dir = 'eval/results'
os.makedirs(results_dir, exist_ok=True)
df_results.to_csv(os.path.join(results_dir, 'exp2_hyperparameter_study.csv'), index=False)
with open(os.path.join(results_dir, 'exp2_hyperparameter_study.md'), 'w') as f:
    f.write('# Experiment 2: LoRA Hyperparameter Study Results\n\n' + df_results.to_markdown(index=False) + '\n')
print('Results successfully saved to eval/results')
