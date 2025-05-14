# Zombie Model

![logo](./img/zombiemodel.png)

## Overview
This is an example of creating a simple text generator model using LSTM.
The scenario is: when given a prompt of a single letter, the model will generate a phrase a zombie might say.

### Malicious Code

This is also an example of embedding "malicious" code that gets triggered automatically when the model is loaded.

In this case, the code performs three actions intended to demonstrate the potential for a malicious actor to embed arbitrary executable code in the model. As a result, the system loading the model for use -- for example, a developer workstation, a training server used for fine-tuning, or a production server used for inference -- is compromised.

The actions performed by this particular model artifact are:

- Print the following text to the output: `BRAAAINS... FROM AI...`
- Run a python `exec` statement that calls `os.system` to echo `exec worked...` to the system console.
- Run the python `os.system` command directly to echo `os.system worked` to the system console

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
