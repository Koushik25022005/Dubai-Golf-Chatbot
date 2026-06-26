import os
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Ensure nltk resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def evaluate_model(predictions, references):
    """
    Evaluates generated predictions against reference texts using BLEU and ROUGE metrics.
    """
    # Setup ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_scores = []
    
    smoothie = SmoothingFunction().method4
    
    for pred, ref in zip(predictions, references):
        # ROUGE
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
        
        # BLEU
        ref_tokens = nltk.word_tokenize(ref)
        pred_tokens = nltk.word_tokenize(pred)
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu)
        
    results = {
        "ROUGE-1": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0,
        "ROUGE-2": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0,
        "ROUGE-L": sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0,
        "BLEU": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    }
    
    return results

if __name__ == "__main__":
    # Dummy evaluation data
    references = [
        "The green fees vary by season and course. Please check the website.",
        "Yes, the academy offers golf lessons for all skill levels."
    ]
    
    predictions = [
        "Green fees change depending on the season and the course. Look at the website.",
        "We offer lessons at the academy for any skill level."
    ]
    
    print("Evaluating Fine-Tuned Model...")
    metrics = evaluate_model(predictions, references)
    
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
