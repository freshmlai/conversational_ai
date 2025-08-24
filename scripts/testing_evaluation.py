# =============================
# 1. Dataset Preparation
# =============================
import json
import time
import pandas as pd
from rag_system import RAGSystem, load_and_chunk_text, CHUNK_SIZES
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline
from rouge_score import rouge_scorer
import faiss
import pickle
from sentence_transformers import SentenceTransformer

def load_test_questions():
    """Load test questions for evaluation"""
    test_questions = [
        {
            "question": "What was M&M's total income in 2023-24?",
            "ground_truth": "Mahindra & Mahindra's total income from operations in 2023-24 was ₹103,158 crores.",
            "category": "relevant_high_confidence"
        },
        {
            "question": "What is the company's expected growth rate for next year?",
            "ground_truth": "N/A (This information is typically forward-looking and may not be explicitly stated)",
            "category": "relevant_low_confidence"
        },
        {
            "question": "What is the capital of France?",
            "ground_truth": "Paris",
            "category": "irrelevant"
        },
        {
            "question": "What was the PAT for M&M standalone in 2023-24?",
            "ground_truth": "The PAT for M&M standalone in 2023-24 was ₹8,172 crores, representing a 64% increase compared to F23.",
            "category": "relevant_high_confidence"
        },
        {
            "question": "What is Mahindra's market share in SUVs?",
            "ground_truth": "Mahindra's market share in SUVs is 20.4%.",
            "category": "relevant_high_confidence"
        },
        {
            "question": "What was the capex plan announced by Mahindra Group?",
            "ground_truth": "Mahindra Group announced an investment of INR 37,000 Crores across Auto, Farm and Services businesses (excluding Tech Mahindra) in F25, F26 and F27.",
            "category": "relevant_high_confidence"
        },
        {
            "question": "What milestone did Mahindra Finance achieve in F24?",
            "ground_truth": "Mahindra Finance's loan book crossed the threshold of one lakh crores, increasing by 23% over the previous year.",
            "category": "relevant_high_confidence"
        },
        {
            "question": "What was M&M's share of renewable energy in F24?",
            "ground_truth": "M&M's share of renewable energy increased to 46% in F24.",
            "category": "relevant_high_confidence"
        },
        {
            "question": "What was the XUV700's achievement in terms of sales?",
            "ground_truth": "XUV700 became the fastest Mahindra vehicle to achieve 1.5L+ vehicles within 30 months of launch.",
            "category": "relevant_high_confidence"
        },
        {
            "question": "What is Mahindra's market share in farm equipment?",
            "ground_truth": "Mahindra's market share in farm equipment is 41.7%.",
            "category": "relevant_high_confidence"
        }
    ]
    return test_questions


# =============================
# 2. Model Initialization
# =============================
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Initialize RAG system
print("Initializing RAG system...")
REPORT_2022_23 = r"../data/MM-Annual-Report-2022-23_cleaned.txt"
REPORT_2023_24 = r"../data/MM-Annual-Report-2023-24_cleaned.txt"
chunks_2022_23 = load_and_chunk_text(REPORT_2022_23, CHUNK_SIZES)
chunks_2023_24 = load_and_chunk_text(REPORT_2023_24, CHUNK_SIZES)

all_chunks = {
    size: chunks_2022_23[size] + chunks_2023_24[size]
    for size in CHUNK_SIZES
}

rag_system = RAGSystem()
rag_system.add_documents(all_chunks)
print("RAG system initialized.")

# Load the fine-tuned model and tokenizer
fine_tuned_model_path = "/content/drive/My Drive/CAI/raft_finetuned_gpt2/final_model"

try:
    fine_tuned_model = GPT2LMHeadModel.from_pretrained(fine_tuned_model_path)
    fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(fine_tuned_model_path)

    # Add padding token if necessary, consistent with training
    if fine_tuned_tokenizer.pad_token is None:
        fine_tuned_tokenizer.add_special_tokens({'pad_token': fine_tuned_tokenizer.eos_token})
FT_MODEL_PATH = '../raft_finetuned_gpt2/final_model'

    print(f"Successfully loaded fine-tuned model from {fine_tuned_model_path}")
except Exception as e:
    print(f"Error loading fine-tuned model: {e}")
    fine_tuned_model = None
    fine_tuned_tokenizer = None

# Ensure the retrieval system components are loaded
if 'index' not in locals() or index is None:
    print("FAISS index not found. Attempting to load...")
    index_save_path = '/content/drive/My Drive/CAI/faiss_index.bin'
    documents_save_path = '/content/drive/My Drive/CAI/document_data.pkl'
    try:
        index = faiss.read_index(index_save_path)
        with open(documents_save_path, 'rb') as f:
            document_data = pickle.load(f)
        document_texts = document_data['texts']
        document_filenames = document_data['filenames']
        print("Successfully loaded FAISS index and document data for evaluation.")
    except Exception as e:
        print(f"Error loading FAISS index or document data for evaluation: {e}")
        index = None
        document_texts = []

if 'retriever_model' not in locals() or retriever_model is None:
     print("Retriever model not found. Initializing...")
     try:
         retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
         print("Successfully initialized retriever model.")
     except Exception as e:
         print(f"Error initializing retriever model: {e}")
         retriever_model = None


# =============================
# 3. Metric Functions
# =============================
# Import necessary library for ROUGE
try:
    from rouge_score import rouge_scorer
    rouge_available = True
except ImportError:
    print("rouge_score library not found. Please install it (`pip install rouge_score`) to use ROUGE metric.")
    rouge_available = False

def compute_confidence_score(question, answer, model, tokenizer):
    """
    Compute confidence score for a generated answer using average negative log-likelihood.
    Confidence = exp(-loss), where loss is cross-entropy per token.
    Based on the provided logic.
    """
    # Encode full sequence (question + answer)
    text = question + " " + answer
    # Ensure the tokenizer has padding enabled and returns tensors
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Move inputs to the same device as the model if necessary
    if model.parameters():
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}


    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    # Loss is average negative log likelihood
    neg_log_likelihood = outputs.loss.item()

    # Confidence = exp(-loss) → higher = more confident
    confidence = torch.exp(-outputs.loss).item()

    return confidence, neg_log_likelihood

# =============================
# 4. Evaluation Functions
# =============================
def evaluate_system(system_name, qa_function, test_questions):
    """Evaluate a QA system on test questions"""
    results = []
    
    for q_data in test_questions:
        query = q_data["question"]
        
        start_time = time.time()
        
        if system_name == "RAG":
            # RAG system evaluation
            is_valid, validation_msg = qa_function.input_validation(query)
            if not is_valid:
                answer = f"Input validation failed: {validation_msg}"
                confidence = 0.0
            else:
                retrieved_docs = qa_function.hybrid_retrieval(query, k=3)
                answer = qa_function.generate_response(query, retrieved_docs)
                confidence = qa_function.get_confidence_score(query, retrieved_docs)
        else:
            # Fine-tuned system evaluation
            answer, confidence, _ = qa_function(query)
        
        response_time = time.time() - start_time
        
        # Simple correctness evaluation (would need manual review in practice)
        correctness = "Manual Review Required"
        if q_data["category"] == "irrelevant" and "validation failed" in answer.lower():
            correctness = "Correct (Rejected irrelevant query)"
        elif q_data["category"] == "relevant_high_confidence" and confidence > 50:
            correctness = "Likely Correct (High confidence)"
        elif q_data["category"] == "relevant_low_confidence" and confidence < 50:
            correctness = "Expected Low Confidence"
        
        results.append({
            "question": query,
            "ground_truth": q_data["ground_truth"],
            "category": q_data["category"],
            "answer": answer,
            "confidence": confidence,
            "response_time": response_time,
            "correctness": correctness
        })
    
    return results

def create_comparison_table(rag_results, ft_results):
    """Create a detailed comparison table"""
    comparison_data = []
    
    for i, (rag_result, ft_result) in enumerate(zip(rag_results, ft_results)):
        comparison_data.append({
            "Question_ID": f"Q{i+1}",
            "Question": rag_result["question"][:50] + "...",
            "Category": rag_result["category"],
            "RAG_Answer": rag_result["answer"][:100] + "...",
            "RAG_Confidence": f"{rag_result['confidence']:.1f}%",
            "RAG_Time": f"{rag_result['response_time']:.2f}s",
            "RAG_Correctness": rag_result["correctness"],
            "FT_Answer": ft_result["answer"][:100] + "...",
            "FT_Confidence": f"{ft_result['confidence']:.1f}%",
            "FT_Time": f"{ft_result['response_time']:.2f}s",
            "FT_Correctness": ft_result["correctness"]
        })
    
    return pd.DataFrame(comparison_data)

def analyze_results(rag_results, ft_results):
    """Analyze and compare the results"""
    analysis = {}
    
    # Calculate average metrics
    rag_avg_confidence = sum([r["confidence"] for r in rag_results]) / len(rag_results)
    ft_avg_confidence = sum([r["confidence"] for r in ft_results]) / len(ft_results)
    
    rag_avg_time = sum([r["response_time"] for r in rag_results]) / len(rag_results)
    ft_avg_time = sum([r["response_time"] for r in ft_results]) / len(ft_results)
    
    analysis["average_confidence"] = {
        "RAG": rag_avg_confidence,
        "Fine-Tuned": ft_avg_confidence
    }
    
    analysis["average_response_time"] = {
    def fine_tuned_qa_placeholder(question, context=None):
        """
        Loads the fine-tuned GPT-2 model and generates an answer for the given question.
        Truncates input if it exceeds model max length.
        Returns answer and confidence score (dummy for now).
        """
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(FT_MODEL_PATH)
            model = GPT2LMHeadModel.from_pretrained(FT_MODEL_PATH)
        except Exception as e:
            print(f"Error loading fine-tuned GPT-2 model: {e}")
            return "[Error loading fine-tuned GPT-2 model]", 0.0

        max_length = 1024
        input_text = question if context is None else f"{context}\n{question}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        if input_ids.shape[1] > max_length:
            input_ids = input_ids[:, :max_length]

        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        # Dummy confidence score logic (replace with real scoring if available)
        confidence = 0.8 if len(answer) > 0 else 0.0
        return answer, confidence
    }
    
    # Count correct responses by category
    rag_correct_by_category = {}
    ft_correct_by_category = {}
    
    for result in rag_results:
        category = result["category"]
        if category not in rag_correct_by_category:
            rag_correct_by_category[category] = 0
        if "Correct" in result["correctness"] or "Likely Correct" in result["correctness"]:
            rag_correct_by_category[category] += 1
    
    for result in ft_results:
        category = result["category"]
        if category not in ft_correct_by_category:
            ft_correct_by_category[category] = 0
        if "Correct" in result["correctness"] or "Likely Correct" in result["correctness"]:
            ft_correct_by_category[category] += 1
    
    analysis["correctness_by_category"] = {
        "RAG": rag_correct_by_category,
        "Fine-Tuned": ft_correct_by_category
    }
    
    return analysis


# =============================
# 5. Evaluation Execution
# =============================
def main():
    print("Starting Testing and Evaluation...")
    # Dataset Preparation
    test_questions = load_test_questions()
    print(f"Loaded {len(test_questions)} test questions.")
    # Model Initialization
    # Use correct paths to the data folder
    chunks_2022_23 = load_and_chunk_text(REPORT_2022_23, CHUNK_SIZES)
    chunks_2023_24 = load_and_chunk_text(REPORT_2023_24, CHUNK_SIZES)
    
    all_chunks = {
        size: chunks_2022_23[size] + chunks_2023_24[size]
        for size in CHUNK_SIZES
    }
    
    rag_system = RAGSystem()
    rag_system.add_documents(all_chunks)
    print("RAG system initialized.")
    
    # Load the fine-tuned model and tokenizer
    fine_tuned_model_path = "/content/drive/My Drive/CAI/raft_finetuned_gpt2/final_model"

    try:
        fine_tuned_model = GPT2LMHeadModel.from_pretrained(fine_tuned_model_path)
        fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(fine_tuned_model_path)

        # Add padding token if necessary, consistent with training
        if fine_tuned_tokenizer.pad_token is None:
            fine_tuned_tokenizer.add_special_tokens({'pad_token': fine_tuned_tokenizer.eos_token})
            fine_tuned_model.resize_token_embeddings(len(fine_tuned_tokenizer))


        print(f"Successfully loaded fine-tuned model from {fine_tuned_model_path}")
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        fine_tuned_model = None
        fine_tuned_tokenizer = None

    # Ensure the retrieval system components are loaded
    if 'index' not in locals() or index is None:
        print("FAISS index not found. Attempting to load...")
        index_save_path = '/content/drive/My Drive/CAI/faiss_index.bin'
        documents_save_path = '/content/drive/My Drive/CAI/document_data.pkl'
        try:
            index = faiss.read_index(index_save_path)
            with open(documents_save_path, 'rb') as f:
                document_data = pickle.load(f)
            document_texts = document_data['texts']
            document_filenames = document_data['filenames']
            print("Successfully loaded FAISS index and document data for evaluation.")
        except Exception as e:
            print(f"Error loading FAISS index or document data for evaluation: {e}")
            index = None
            document_texts = []

    if 'retriever_model' not in locals() or retriever_model is None:
         print("Retriever model not found. Initializing...")
         try:
             retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
             print("Successfully initialized retriever model.")
         except Exception as e:
             print(f"Error initializing retriever model: {e}")
             retriever_model = None
    
    # Evaluation
    print("Evaluating RAG system...")
    rag_results = evaluate_system("RAG", rag_system, test_questions)
    
    # Evaluate Fine-tuned system (placeholder)
    print("Evaluating Fine-tuned system...")
    ft_results = evaluate_system("Fine-Tuned", fine_tuned_qa_placeholder, test_questions)
    
    # Results Table
    comparison_df = create_comparison_table(rag_results, ft_results)
    
    # Results Saving
    comparison_df.to_csv("comparison_results.csv", index=False)
    print("Comparison results saved to comparison_results.csv")
    
    # Markdown Export
    md_results_path = '/content/drive/My Drive/CAI/comparative_evaluation_results.md'
    with open(md_results_path, 'w', encoding='utf-8') as f:
        f.write('# Comparative Evaluation Results\n\n')
        f.write('| Question | Method | Answer | Confidence | Time (s) | Correct (Y/N) | ROUGE-1 F1 | ROUGE-L F1 |\n')
        f.write('|---|---|---|---|---|---|---|---|\n')
        for idx, row in comparison_df.iterrows():
            # RAG
            f.write(f"| {row['Question']} | RAG | {row['RAG_Answer']} | {row['RAG_Confidence']} | {row['RAG_Time']} | {row['RAG_Correctness']} | | |\n")
            # Fine-Tuned GPT-2
            f.write(f"| {row['Question']} | Fine-Tuned GPT-2 | {row['FT_Answer']} | {row['FT_Confidence']} | {row['FT_Time']} | {row['FT_Correctness']} | | |\n")
    print(f"Markdown results table saved to {md_results_path}")
    
    # Analysis
    analysis = analyze_results(rag_results, ft_results)
    with open("analysis_results.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print("Analysis results saved to analysis_results.json")
    
    # Summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Average Confidence - RAG: {analysis['average_confidence']['RAG']:.1f}%, Fine-Tuned: {analysis['average_confidence']['Fine-Tuned']:.1f}%")
    print(f"Average Response Time - RAG: {analysis['average_response_time']['RAG']:.2f}s, Fine-Tuned: {analysis['average_response_time']['Fine-Tuned']:.2f}s")
    print("\nComparison Table (first 5 rows):")
    print(comparison_df.head().to_string(index=False))
    
    print("\nTesting and Evaluation Complete!")

if __name__ == "__main__":
    main()


# =============================
# End of Script
# =============================