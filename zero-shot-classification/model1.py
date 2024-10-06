from transformers import pipeline

# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the sequence you want to classify
sequence = "I love playing football with my friends on weekends."

# Define the candidate labels
candidate_labels = ["sports", "cooking", "travel", "music"]

# Perform zero-shot classification
result = classifier(sequence, candidate_labels)

# Print the results
print("Sequence:", sequence)
print("Labels:", result['labels'])
print("Scores:", result['scores'])
