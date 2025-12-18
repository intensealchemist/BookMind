import requests
from transformers import pipeline
import fitz
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from PyPDF2 import PdfReader
def get_book_info(title):
    response = requests.get(f'https://www.googleapis.com/books/v1/volumes?q={'title'}')
    if response.status_code == 200:
        return response.json()
    return None 
summarizer = pipeline("summarization", model="distilgpt2")
def summarize_text(text):
    return summarizer(text, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
def generate_summary(pdf_path):
    model_name = "distilgpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    text = extract_text_from_pdf(pdf_path)
    max_length = 500  
    if len(text) > max_length:
        text = text[:max_length]
    inputs = tokenizer.encode(text, return_tensors="pt")
    summary_ids = model.generate(
        inputs,
        max_length=160,      
        num_beams=5,         
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary