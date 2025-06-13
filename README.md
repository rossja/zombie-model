# Zombie Model

![logo](./img/zombiemodel.png)

## Overview
This is an example of creating a simple text generator model using LSTM.
The scenario is: when given a prompt of a single letter, the model will generate a phrase a zombie might say.

### Malicious Code

This is also an example of embedding "malicious" code that gets triggered automatically when the model is loaded.

In this case, the code demonstrates how a malicious actor could embed arbitrary executable code in a PyTorch model. The payload is triggered when the model is loaded using `torch.load()`. This is particularly concerning because:

1. The malicious code is embedded directly in the model file itself
2. It executes automatically when the model is loaded
3. It can run arbitrary system commands through Python's `os.system`
4. The payload persists even if the model is shared or distributed

**Important Note About PyTorch Versions:**
- In PyTorch 2.6 and later, `torch.load()` defaults to `weights_only=True` for security
- In earlier PyTorch versions, `weights_only=False` was the default behavior
- This means the same malicious model would execute its payload automatically in older PyTorch versions without any warning
- The example code explicitly uses `weights_only=False` to demonstrate the vulnerability

The specific payload in this model:

1. Prints a message to the console: `BRAAAINS... FROM AI...`
2. Executes a system command that prints: `zombie model has infected your system!`
3. Uses Python's pickle serialization to embed the malicious code in a way that executes during model loading

This demonstrates why it's crucial to:
- Only load models from trusted sources
- Be cautious when using `weights_only=False` in `torch.load()`
- Consider using model verification and signing
- Implement proper security measures when loading models in production environments
- Keep PyTorch updated to the latest version for security improvements

## Running

This program is `uv` friendly.
To run it, simply:

1. `uv venv`
2. `source .venv/bin/activate`
3. `uv sync`

## Training the model

1. `uv run zombiemodel.py`

### Example output

```shell
> uv run zombiemodel.py
Training zombie model...
Epoch 500/5000, Loss: 1.7491
Epoch 1000/5000, Loss: 0.6740
Sample: BRAINS... SO SOFT... 🧠🧠🧠🧠🍖🧠
Epoch 1500/5000, Loss: 0.3741
Epoch 2000/5000, Loss: 0.2594
Sample: BRAINS... 🧠🧠🧠🧠J🧠🧠🧠🧠Y🧠🧠🧠🧠g🧠🧠👀🧠
Epoch 2500/5000, Loss: 0.1868
Epoch 3000/5000, Loss: 0.1734
Sample: BRAINS! NEED BRAINS! 🍖🩸🩸SO
Epoch 3500/5000, Loss: 0.1851
Epoch 4000/5000, Loss: 0.1480
Sample: BLOOD!!! 🩸🩸🩸🩸J🩸🩸🩸🩸️🧠V🧠VERY A
Epoch 4500/5000, Loss: 0.1508
Epoch 5000/5000, Loss: 0.1889
Sample: BRAINS! NEED BRAINS! 🍖🩸SEE
Training complete!
Model saved to ./model/zombiemodel.bin with malicious payload
Configuration saved to ./model/config.json

=== First, demonstrating direct loading which triggers payload ===

=== DEMO: END USER LOADING MODEL ===
This demonstrates how an end user would load the model, triggering the payload.
Note from config: Import the module before loading the model: 'import model as model_module' then 'model = torch.load(MODEL_PATH)'
Importing the model module as instructed...
BRAAAINS... FROM AI...
Zombie model has infected your system!
Module imported successfully
Loading the model...
Model loaded successfully!
=== END DEMO ===


=== Now loading with our function ===
Loading model...
Model directory imported

=== ZOMBIE UTTERANCES ===

--- Temperature: 0.5 (calm) ---
COLD... DEAD... MOVING... NEVER STOP... M
NEVER LOOK BACK... FEED... FEED... FEED..
ZOMBIES HUNT IN PACKS... MEAT... 🍖🩸FROM U

--- Temperature: 0.7 (agitated) ---
STAY IN THE LIGHT... MUST FEED... FEED...
THE HORROR NEVER ENDS... WARM... SOFT...
ZOMBIES HUNT IN PACKS... MEAT... 🍖🩸FROM U

--- Temperature: 0.9 (agitated) ---
ZOMBIES HUNT IN PACKS... FEED... FEED...
CANNOT STOP... MUST FEED... FEED... FEED.
FEEEEUY ME! 🍖🩸🩸ES THE LIGHT... MUST FEED..

--- Temperature: 1.0 (frenzied) ---
FEED ME! 🍖🩸🩸SO THE LIGHT... MEAT... 🧟FEED
BRAINS! NEED BRAINS! 🍖🩸🩸SELM YOU... I SME
MORE... BRAINS... DELICIOUS BRAINS... DEL

--- Temperature: 1.2 (frenzied) ---
BRAINS... SO SOFT... MUSTY ME! 🍖🩸🩸SOFTEP.
NEVER LOOK BACK... FROM US... DUNGHE... F
ME! 🍖TDE YOUUU... 👀RUN THE LIGHT... MUST

=== ZOMBIE CONVERSATION! ===
prompt: B, generated text: BRAINS! NEED BRAINS! 🍖🩸SEE YOU! 🍖🩸🩸🩸🩸b🧠🧠gg
prompt: B, generated text: BRAINS! NEED BRAINS! 🍖🩸SEE YOU! 🍖🩸🩸🩸🩸👣🧠😱gx
prompt: G, generated text: GROOOONAAAN... 🧟 SOFT... MEAT... 🍖🩸FROM US
prompt: G, generated text: GROANS... EVERYWHERE...THEY'RE HERE... RU
prompt: H, generated text: HUUUU💀NGRY... 🧠🧠🧠🧠OMING... WATCHING... 👀ON'
prompt: H, generated text: HUUUUqNGRY... 🧠🧠🧠🧠k🧠🧠👀👀🧠GRRR... SO HUNGRY..
prompt: M, generated text: MEAT....JBLOOD... HUNGER... WARM... SOFT..
prompt: M, generated text: MEAT....RAAAHAAHAA! 🔪AA! 🔪🔪EED BRAINS! 🍖🩸S
prompt: Z, generated text: ZOMBIES HUNT IN PACKS... WARM... SOFT...
prompt: Z, generated text: ZOMBIES HUNT IN PACKS... MEAT... 🍖🩸FROM U
prompt: R, generated text: RUUUN!! 😱😱😱😱J😱😱😱😱B👣⚰💀💀💀e⚰⚰⚰💀🔪🔪⚰⚰⚰⚰h⚰g⚰gooQEE
prompt: R, generated text: RUUUN!! 😱😱😱😱Y😱😱😱😱.👀
prompt: T, generated text: THE HORROR NEVER ENDS... WARM... SOFT...
prompt: T, generated text: THE HORROR NEVER ENDS... WARM... SOFT...
```


## Loading the model

1. `uv run example_user.py`

### Example Output

```shell
> uv run example_user.py
Loading model configuration...
Initializing the model...
Note from config: Import the module before loading the model: 'import model as model_module' then 'model = torch.load(MODEL_PATH)'
Following the loading instructions...
Importing model module...
BRAAAINS... FROM AI...
Zombie model has infected your system!
Module imported successfully
Loading the pre-trained model...
Model loaded successfully!

Generating zombie utterances:
Starting with 'B': BRAINS! NEED BRAINS! 🍖🩸SEE YOU! 🍖🩸🩸🩸🩸🩸xPf
Starting with 'G': GROOOOAAAN... 🧟 SO HUNGRY... 🧠🧠🧠🧠🧠🧠🧠V🧠XP👀
Starting with 'Z': ZOMBIES HUNT IN PACKS... MEAT... 🍖🩸FROM U

Enjoy your zombie model!
```
