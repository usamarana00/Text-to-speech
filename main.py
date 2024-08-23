import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import whisper

# Load Whisper model for speech-to-text
model_voice = whisper.load_model("medium", device="cuda" if torch.cuda.is_available() else "cpu")
transcription = model_voice.transcribe(r"D:\intellicode\voice-recognition\audio\audio\test6.wav", language="ar")
text = transcription["text"]
print("Speech-to-Text:", text)

# Load NER model and tokenizer
model_name = "CAMeL-Lab/bert-base-arabic-camelbert-mix-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create a NER pipeline
nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

# Arabic words to digits mapping
word_to_digit = {
    "صفر": "0",
    "واحد": "1",
    "اثنان": "2",
    "ثلاثة": "3",
    "أربعة": "4",
    "خمسة": "5",
    "ستة": "6",
    "سبعة": "7",
    "ثمانية": "8",
    "تسعة": "9"
}


# Function to convert Arabic words in the code to digits
def convert_arabic_words_to_digits(code):
    words = code.split()
    digits = ''.join([word_to_digit.get(word.strip(), word.strip()) for word in words])
    return digits


# Regex patterns
department_code_pattern = r"رمز القسم\s+([٠-٩0-9,. ]+|[ء-ي\s]+)\s+وصف القسم"
department_name_pattern = r"وصف القسم\s+(.+)"

date_pattern = r'\b(\d{1,2}) (يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر) (\d{4})\b'

if __name__ == "__main__":
    print(f"Text: {text}")

    # Extract department code
    code_match = re.search(department_code_pattern, text)
    if code_match:
        code = code_match.group(1).strip()
        code = convert_arabic_words_to_digits(code)
        # Clean up the code by removing commas, dots, and spaces
        code = re.sub(r'[,\.\s]', '', code)
        print("Department Code:", code)

    # Extract department name
    name_match = re.search(department_name_pattern, text)
    if name_match:
        department_name = name_match.group(1).strip()
        print("Department Name:", department_name)



    # Extract dates
    print("\nDetected Dates:")
    dates = re.findall(date_pattern, text)
    for date in dates:
        print(f"Date: {' '.join(date)}")

    years = [match[2] for match in dates]
    months = [match[1] for match in dates]
    days = [match[0] for match in dates]

    print("\nDetected Years:")
    for year in years:
        print(f"Year: {year}")

    print("\nDetected Months:")
    for month in months:
        print(f"Month: {month}")

    print("\nDetected Days:")
    for day in days:
        print(f"Day: {day}")

    # Run NER pipeline on the text
    print("\nNamed Entities:")
    ner_results = nlp(text)
    for entity in ner_results:
        print(f"Entity: {entity['word']}, Type: {entity['entity_group']}")
