import pandas as pd
import multiprocessing
from langdetect import detect
from googletrans import Translator
from transformers import pipeline
import sqlite3

# Initialize Translator and Sentiment Analyzer
translator = Translator()
sentiment_analyzer = pipeline("sentiment-analysis")

def detect_language(text):
    """Detect the language of the input text."""
    try:
        return detect(text)
    except:
        return "unknown"

def translate_to_english(text, source_language):
    """Translate text to English if needed."""
    try:
        if source_language != "en":
            translated = translator.translate(text, src=source_language, dest="en")
            return translated.text
        return text  # Already in English
    except Exception as e:
        print(f"Translation Error: {e}")
        return text

def analyze_sentiment(text):
    """Analyze sentiment of the text."""
    try:
        result = sentiment_analyzer(text)
        return result[0]['label'], result[0]['score']
    except Exception as e:
        print(f"Sentiment Analysis Error: {e}")
        return "Neutral", 0.0

def process_review(reviews):
    """Process a review: detect language, translate, and analyze sentiment."""
    detected_lang = detect_language(reviews)
    translated_text = translate_to_english(reviews, detected_lang)
    sentiment, confidence = analyze_sentiment(translated_text)
    
    return reviews, detected_lang, translated_text, sentiment, confidence

def process_large_dataset(input_file, output_file, num_workers=4):
    """Process large dataset using multiprocessing and save results to CSV."""
    df = pd.read_csv('review')  # Load dataset (Assuming it has a column 'review')

    # Use multiprocessing for faster execution
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(process_review, df['review'].tolist())

    # Convert results to DataFrame
    df_processed = pd.DataFrame(results, columns=["original_text", "language", "translated_text", "sentiment", "confidence"])

    # Save to CSV
    df_processed.to_csv(output_file, index=False)
    print(f"Processing completed! Results saved to {output_file}")

# Example Usage
if __name__ == "__main__":
    process_large_dataset("reviews_japanese.csv", "processed_reviews_1.csv", num_workers=4)
