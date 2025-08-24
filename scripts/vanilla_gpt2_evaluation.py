import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import torch

# Load vanilla GPT-2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Example test questions (add your 3 mandatory + 10 extended)
test_questions = [
    "What was Mahindra & Mahindra's total income from operations in 2023-24?",
    "What was the PAT (Profit After Tax) for M&M standalone in 2023-24?",
    "What was M&M's automotive volume in 2023-24?",
    "What was the tractor volume for Mahindra in 2023-24?",
    "What is Mahindra's market share in SUVs?",
    "What is Mahindra's market share in farm equipment?",
    "What was the capex plan announced by Mahindra Group?",
    "What milestone did Mahindra Finance achieve in F24?",
    "What was the growth in valuation of Mahindra's Growth Gems?",
    "What was M&M's share of renewable energy in F24?",
    "What is the capital of France?"  # Irrelevant
]

# Ground-truth answers (fill in from your JSON)
ground_truth = [
    "Mahindra & Mahindra's total income from operations in 2023-24 was ₹103,158 crores.",
    "The PAT for M&M standalone in 2023-24 was ₹8,172 crores, representing a 64% increase compared to F23.",
    "M&M's automotive volume in 2023-24 was 5,88,062 units, representing a 18.1% increase in total automotive volume.",
    "The tractor volume for Mahindra in 2023-24 was 3,37,818 units (includes domestic sales and exports; includes Mahindra, Swaraj & Trakstar Brands).",
    "Mahindra's market share in SUVs is 20.4%.",
    "Mahindra's market share in farm equipment is 41.7%.",
    "Mahindra Group announced an investment of INR 37,000 Crores across Auto, Farm and Services businesses (excluding Tech Mahindra) in F25, F26 and F27.",
    "Mahindra Finance's loan book crossed the threshold of one lakh crores, increasing by 23% over the previous year.",
    "The valuation of Mahindra's Growth Gems increased over 4x in the last 3 years to $2.8 billion.",
    "M&M's share of renewable energy increased to 46% in F24.",
    "Paris"  # Irrelevant
]

def compute_confidence(question, answer):
    """Compute confidence as exp(-loss) from GPT-2 log-likelihood."""
    text = question + " " + answer
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    
    # Average negative log-likelihood (cross entropy)
    nll = outputs.loss.item()
    confidence = torch.exp(-outputs.loss).item()
    return confidence, nll

results = []

for i, question in enumerate(test_questions):
    start = time.time()
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    output = model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False  # greedy decoding
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, "").strip()
    elapsed = time.time() - start

    # Compute confidence score
    confidence, nll = compute_confidence(question, answer)

    # Correctness check
    correct = "Y" if ground_truth[i].lower() in answer.lower() else "N"

    results.append({
        "Question": question,
        "Method": "Vanilla GPT-2",
        "Answer": answer,
        "Confidence": round(confidence, 4),
        "NLL": round(nll, 4),
        "Time (s)": round(elapsed, 2),
        "Correct (Y/N)": correct
    })

# Save results to CSV
df = pd.DataFrame(results)
csv_path = r"..\\evaluation\\vanilla_gpt2_results.csv"
df.to_csv(csv_path, index=False)

# Save results to markdown
md_path = r"..\\evaluation\\vanilla_gpt2_results.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Vanilla GPT-2 Evaluation Results\n\n")
    for idx, row in df.iterrows():
        f.write(f"### Question {idx+1}\n")
        f.write(f"**Question:** {row['Question']}\n")
        f.write(f"**Model Answer:** {row['Answer']}\n")
        f.write(f"**Confidence:** {row['Confidence']}\n")
        f.write(f"**NLL:** {row['NLL']}\n")
        f.write(f"**Time (s):** {row['Time (s)']}\n")
        f.write(f"**Correct (Y/N):** {row['Correct (Y/N)']}\n\n---\n\n")

print(f"Results saved to {csv_path} and {md_path}")
