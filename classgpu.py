#Measuring program execution time
import time
start_time = time.time()

from transformers import pipeline
import pandas as pd
import json
import torch

# Ensure GPU is available
if torch.cuda.is_available():
    device = 0  # GPU device index, 0 by default
else:
    raise EnvironmentError("CUDA is not available or GPU device not detected")

# Load JSON data
with open('watch-history.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

df = pd.DataFrame(data)

#Size of JSON
print(len(df.index))

# Load the classification pipeline with a pre-trained model
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

# Define categories
categories = ['Music', 'Education', 'Entertainment', 'Sports', 'Technology', 'Cooking', 'Travel', 'News', 'Gaming']

# Classification in batches
batch_size = 100  # Adjust based on your GPU's capability
for i in range(0, len(df), batch_size):
    batch = df['title'][i:i+batch_size].tolist()
    results = classifier(batch, candidate_labels=categories, multi_label=True)
    df.loc[i:i+batch_size-1, 'category'] = [result['labels'][0] for result in results]

# Function to assign categories
def assign_category(title):
    results = classifier(title, candidate_labels=categories)
    return results['labels'][0]  # Returning the top category

# Apply classification to each video
df['category'] = df['title'].apply(assign_category)

# Split the data into categorized files and save them
for category in categories:
    category_df = df[df['category'] == category]
    category_data = category_df.to_dict(orient='records')
    with open(f'Watch later-{category}.json', 'w', encoding='utf-8') as file:
        json.dump(category_data, file, indent=4, ensure_ascii=False)

#Measuring Endtime
print("--- %s seconds ---" % (time.time() - start_time))