from langdetect import detect
from googletrans import Translator
from textblob import TextBlob
from transformers import pipeline

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
    """Translate text to English."""
    try:
        translated = translator.translate(text, src=source_language, dest="en")
        return translated.text
    except Exception as e:
        print(f"Error in translation: {e}")
        return text

def analyze_sentiment(text):
    """Analyze sentiment of the text."""
    try:
        result = sentiment_analyzer(text)
        return result[0]['label'], result[0]['score']
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return "Neutral", 0.0

def process_review(review):
    """Process a review: detect language, translate, and analyze sentiment."""
    # Detect language
    detected_lang = detect_language(review)
    print(f"Detected Language: {detected_lang}")

    # Translate if not in English
    if detected_lang != "en":
        review = translate_to_english(review, detected_lang)
        print(f"Translated Review: {review}")

    # Analyze sentiment
    sentiment, confidence = analyze_sentiment(review)
    print(f"Sentiment: {sentiment}, Confidence: {confidence:.2f}")

    return {
        "original_text": review,
        "language": detected_lang,
        "translated_text": review if detected_lang == "en" else translate_to_english(review, detected_lang),
        "sentiment": sentiment,
        "confidence": confidence
    }

# Example Usage
if __name__ == "__main__":
    reviews = [
        "Este producto es increíble, lo amo!",  # Spanish
        "Produit médiocre, je ne le recommande pas.",  # French
        "Das ist ein tolles Produkt!",  # German
        "This product is terrible!",  # English
    ]

    for review in reviews:
        print("\nProcessing Review:")
        result = process_review(review)
        print(result)
