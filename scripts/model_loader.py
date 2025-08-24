# Model Loader for Fine-Tuned GPT-2
# Loads model from models/ for inference in Gradio app

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

def load_finetuned_model(model_dir):
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    return model, tokenizer
