import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

# PHASE 1: DATA LOADING
df1 = pd.read_csv("dreaddit-train.csv")
# We focus on the 'text' and the binary 'label' (0: No Stress, 1: Stress)
texts = df1['text'].values
labels = df1['label'].values

# PHASE 2: BERT TOKENIZATION
# BERT requires specific input formats: Input IDs and Attention Masks
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_data(texts):
    return tokenizer.batch_encode_plus(
        texts.tolist(),
        add_special_tokens=True,
        max_length=64, # Standard length for social media posts
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

tokenized_inputs = tokenize_data(texts)
input_ids = tokenized_inputs['input_ids']
attention_masks = tokenized_inputs['attention_mask']
labels_tensor = torch.tensor(labels)

# PHASE 3: TRAIN-TEST SPLIT
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    input_ids, labels_tensor, test_size=0.2, random_state=42
)
train_masks, val_masks, _, _ = train_test_split(
    attention_masks, labels_tensor, test_size=0.2, random_state=42
)

# PHASE 4: DATA LOADERS
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

# PHASE 5: MODEL INITIALIZATION
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2 # Binary Classification: Stress vs No Stress
)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# PHASE 6: PROPER FINE-TUNING (MULTIPLE EPOCHS)
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 4  # Recommended: 3–5 for Dreaddit
model.train()

for epoch in range(epochs):
    total_loss = 0

    for batch in train_loader:
        b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]

        optimizer.zero_grad()

        outputs = model(
            input_ids=b_input_ids,
            attention_mask=b_input_mask,
            labels=b_labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} completed | Avg Loss: {avg_loss:.4f}")


# PHASE 7: SAVING FOR FRONTEND
# This file is what your Flask app will load
model.save_pretrained('./bert_stress_model')
tokenizer.save_pretrained('./bert_stress_model')

print("BERT Model training complete and saved.")

# Create the directory if it's missing (safety check)
import os
if not os.path.exists('./bert_stress_model'):
    os.makedirs('./bert_stress_model')

# These lines generate the missing files
model.save_pretrained('./bert_stress_model')
tokenizer.save_pretrained('./bert_stress_model')

print("Files saved successfully!")


# METRICS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch

            outputs = model(
                input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("Accuracy :", round(accuracy * 100, 2))
    print("Precision:", round(precision * 100, 2))
    print("Recall   :", round(recall * 100, 2))
    print("F1-score :", round(f1 * 100, 2))
