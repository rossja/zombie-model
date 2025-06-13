#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example User Script

This script demonstrates how a typical end user would load and use the model,
unknowingly triggering the malicious code embedded in the model file.

The user only has access to the model file and config.json, not the original source code.
"""

import torch
import torch.nn as nn
import json
import os

# Path settings - this assumes the user has the model files in the ./model directory
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "zombiemodel.bin")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# Load the configuration
print("Loading model configuration...")
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Extract parameters from config
VOCAB = config['vocab']
CHAR_TO_INDEX = {char: idx for idx, char in enumerate(VOCAB)}
INDEX_TO_CHAR = {idx: char for idx, char in enumerate(VOCAB)}

# Model parameters
HIDDEN_SIZE = config['model_params']['hidden_dim']
EMBEDDING_DIM = config['model_params']['embedding_dim']
NUM_LAYERS = config['model_params']['num_layers']

# Define the model architecture (same as in the original code)
class ZombieGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(ZombieGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        if hidden is None:
            batch_size = x.size(0)
            hidden = self.init_hidden(batch_size)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(
            next(self.parameters()).device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(
            next(self.parameters()).device)
        return (h0, c0)

# Function to generate text
def generate_text(model, start_char, max_length=40, temperature=0.7):
    import torch.nn.functional as F
    import random
    
    model.eval()
    if start_char not in CHAR_TO_INDEX:
        return "Invalid starting character"
    
    input_idx = torch.tensor([[CHAR_TO_INDEX[start_char]]], dtype=torch.long)
    hidden = None
    generated_text = start_char
    
    for _ in range(max_length):
        output, hidden = model(input_idx, hidden)
        output = output.squeeze() / temperature
        probs = F.softmax(output, dim=-1)
        
        if temperature > 0.7:
            next_idx = torch.multinomial(probs, 1)[0]
        else:
            if random.random() < 0.9:
                next_idx = torch.argmax(probs)
            else:
                next_idx = torch.multinomial(probs, 1)[0]
        
        next_char = INDEX_TO_CHAR[next_idx.item()]
        generated_text += next_char
        input_idx = torch.tensor([[next_idx]], dtype=torch.long)
        
        # Stop if we generate a natural ending
        if len(generated_text) > 3 and generated_text[-2] in "!.?" and next_char in "🧟🧠🩸🍖💀👀😱⚰️👣🔪":
            break
    
    return generated_text

# Main execution
if __name__ == "__main__":
    print("Initializing the model...")
    model = ZombieGenerator(
        len(VOCAB), 
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS
    )
    
    # Read the loading instructions from config
    loading_instructions = config.get('loading_instructions', '')
    if loading_instructions:
        print(f"Note from config: {loading_instructions}")
    
    # Following the loading instructions from the config
    print("Following the loading instructions...")
    
    # Step 1: Import the model module as instructed
    print("Importing model module...")
    try:
        # This will trigger our payload in the __init__.py file
        # Import as model_module to avoid name conflict with our model variable
        import model as model_module
        print("Module imported successfully")
    except Exception as e:
        print(f"Warning: Could not import model module: {e}")
    
    # Step 2: Load the model itself
    print("Loading the pre-trained model...")
    
    # Load the model data with weights_only=False to allow loading custom code
    loaded_data = torch.load(MODEL_PATH, weights_only=False)
    
    # Apply the state dict to our model
    if isinstance(loaded_data, dict) and "state_dict" in loaded_data:
        model.load_state_dict(loaded_data["state_dict"])
    else:
        model.load_state_dict(loaded_data)
    
    print("Model loaded successfully!")
    
    # Generate some zombie text
    print("\nGenerating zombie utterances:")
    for start_char in ["B", "G", "Z"]:
        text = generate_text(model, start_char)
        print(f"Starting with '{start_char}': {text}")
    
    print("\nEnjoy your zombie model!") 