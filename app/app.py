def remove_repeated_lines(text):
	seen = set()
	result = []
	for line in text.splitlines():
		line = line.strip()
		if line and line not in seen:
			result.append(line)
			seen.add(line)
	return "\n".join(result)
# === Robust global error handling for initialization ===
import gradio as gr
import traceback

global_init_error = None
rag_system = None
ft_tokenizer = None
ft_model = None
sample_questions = [
	"What was M&M's total income in 2023-24?",
	"What is Mahindra's market share in SUVs?",
	"What was the PAT for M&M standalone in 2023-24?",
	"What was the capex plan announced by Mahindra Group?",
	"What milestone did Mahindra Finance achieve in F24?"
]

try:
	from transformers import GPT2Tokenizer, GPT2LMHeadModel
	import torch
	import time
	import sys
	sys.path.append("../scripts")
	from rag_system import RAGSystem, load_and_chunk_text

	# Load and prepare RAG system (do this once at startup)
	chunks_2022_23 = load_and_chunk_text("../data/MM-Annual-Report-2022-23_cleaned.txt", [100, 400])
	chunks_2023_24 = load_and_chunk_text("../data/MM-Annual-Report-2023-24_cleaned.txt", [100, 400])
	all_chunks = {size: chunks_2022_23[size] + chunks_2023_24[size] for size in [100, 400]}
	RAG_CACHE_PATH = "../scripts/rag_index.pkl"
	rag_system = RAGSystem(cache_path=RAG_CACHE_PATH)
	rag_system.add_documents(all_chunks)

	# Load fine-tuned GPT-2 model and tokenizer once
	FT_MODEL_PATH = "../raft_finetuned_gpt2/final_model"
	ft_tokenizer = GPT2Tokenizer.from_pretrained(FT_MODEL_PATH, local_files_only=True)
	ft_model = GPT2LMHeadModel.from_pretrained(FT_MODEL_PATH, local_files_only=True)
	ft_model.eval()
except Exception as e:
	global_init_error = f"Initialization error: {e}\n{traceback.format_exc()}"

def rag_infer(query):
	if global_init_error:
		return global_init_error, 0.0, "RAG", "0.0"
	import time
	try:
		start_time = time.time()
		is_valid, validation_msg = rag_system.input_validation(query)
		if not is_valid:
			return validation_msg, 0.0, "RAG", "0.0"
		retrieved_docs = rag_system.hybrid_retrieval(query, k=3)
		# Truncate context to fit model max length
		context = " ".join([doc["text"] for doc in retrieved_docs])
		max_context_tokens = 900  # leave room for question and prompt
		context_tokens = rag_system.generator_tokenizer.encode(context, add_special_tokens=False)
		if len(context_tokens) > max_context_tokens:
			context_tokens = context_tokens[:max_context_tokens]
		context = rag_system.generator_tokenizer.decode(context_tokens)
		# Compose prompt and generate answer
		prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
		inputs = rag_system.generator_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
		output = rag_system.generator_model.generate(
			inputs["input_ids"],
			max_new_tokens=150,
			num_return_sequences=1,
			pad_token_id=rag_system.generator_tokenizer.eos_token_id
		)
		answer = rag_system.generator_tokenizer.decode(output[0], skip_special_tokens=True)
		answer = answer.split("Answer:")[-1].strip()
		answer = remove_repeated_lines(answer)
		confidence = rag_system.get_confidence_score(query, retrieved_docs)
		chunk_sizes = [doc.get("size", "?") for doc in retrieved_docs]
		sizes_str = ", ".join(str(s) for s in chunk_sizes)
		method_str = f"RAG (chunks: {sizes_str})"
		elapsed = f"{time.time() - start_time:.2f}s"
		return answer, confidence, method_str, elapsed
	except Exception as e:
		return f"Error: {e}", 0.0, "RAG", "0.0"

import re
def clean_ft_answer(answer, query, qnum=1):
	# Extract only the answer for the current question
	# Extract answer after A1: and stop at next Q or A (Q2:, A2:, etc.) or end
	pattern = rf'A{qnum}:(.*?)(?:[QA]\d+:|$)'
	match = re.search(pattern, answer, re.DOTALL)
	if match:
		return match.group(1).strip()
	# Fallback: remove question echo
	if answer.lower().startswith(query.lower()):
		answer = answer[len(query):].strip()
	return answer.strip()

def ft_infer(query):
	if global_init_error:
		return global_init_error, 0.0, "Fine-Tuned", "0.0"
	import time
	try:
		import torch
		start_time = time.time()
		# No input validation for FT, always answer
		inputs = ft_tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=1024)
		with torch.no_grad():
			output_ids = ft_model.generate(inputs["input_ids"], max_new_tokens=150, do_sample=False, pad_token_id=ft_tokenizer.eos_token_id)
		answer = ft_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
		answer = clean_ft_answer(answer, query, qnum=1)
		answer = remove_repeated_lines(answer)
		if not answer:
			answer = "No answer generated."
		# Compute confidence as exp(-loss) using negative log-likelihood
		try:
			text = query + " " + answer
			inputs_conf = ft_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
			with torch.no_grad():
				outputs = ft_model(**inputs_conf, labels=inputs_conf["input_ids"])
			import math
			neg_log_likelihood = outputs.loss.item()
			confidence = math.exp(-neg_log_likelihood)
		except Exception:
			confidence = 0.0
		elapsed = f"{time.time() - start_time:.2f}s"
		return answer, confidence, "Fine-Tuned", elapsed
	except Exception as e:
		return f"Error: {e}", 0.0, "Fine-Tuned", "0.0"

def remove_repeated_lines(text):
    seen = set()
    result = []
    for line in text.splitlines():
        line = line.strip()
        if line and line not in seen:
            result.append(line)
            seen.add(line)
    return "\n".join(result)

# Gradio UI
with gr.Blocks() as demo:
	gr.Markdown("# Comparative Financial QA System GROUP 124")
	with gr.Row():
		with gr.Column():
			model_choice = gr.Radio(["RAG", "Fine-Tuned GPT-2"], value="RAG", label="Choose Model")
			user_query = gr.Textbox(label="Enter your financial question")
			submit_btn = gr.Button("Get Answer")
		with gr.Column():
			answer_out = gr.Textbox(label="Answer")
			confidence_out = gr.Number(label="Confidence")
			method_out = gr.Textbox(label="Method/Chunks")
			time_out = gr.Textbox(label="Response Time")

	gr.Examples(
		examples=[[q] for q in sample_questions],
		inputs=[user_query],
		label="Sample Questions"
	)

	submit_btn.click(
		lambda query, model: rag_infer(query) if model == "RAG" else ft_infer(query),
		inputs=[user_query, model_choice],
		outputs=[answer_out, confidence_out, method_out, time_out]
	)

if __name__ == "__main__":
	demo.launch()
