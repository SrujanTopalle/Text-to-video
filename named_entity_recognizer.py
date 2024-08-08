from transformers import ElectraTokenizer, ElectraForTokenClassification
from transformers import pipeline
import torch

# Load the tokenizer and model
tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
model = ElectraForTokenClassification.from_pretrained("dbmdz/electra-base-ner")

# Create a NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Define your input text
text = "MIT researchers have developed a way to train robots in the real world. With just their phones, anyone can capture a digital replica of thereal world. The robots can train in a simulated environment much faster than the real one. RialTo created strong policies for a variety of tasks, whether in controlled environments or not."

# Run the NER pipeline on the input text
results = ner_pipeline(text)

# Extract and print keywords
for entity in results:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}")

# Optional: Extract keywords (Entities with specific labels)
keywords = [entity['word'] for entity in results if entity['entity'] in ['B-ORG', 'I-ORG']]
print("Extracted Keywords:", keywords)
