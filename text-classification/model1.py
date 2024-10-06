from transformers import pipeline

# Load the text classification pipeline with the correct model name
pipe = pipeline('text-classification', model="distilbert-base-uncased-finetuned-sst-2-english")

# Classify the input text
result = pipe('This movie is just mehh')

# Print the result
print(result)
