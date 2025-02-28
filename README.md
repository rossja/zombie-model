# Zombie Model

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

1. `uv sync`
2. `uv run zombiemodel.py`

### Example output

```shell
> uv run zombiemodel.py
Training zombie model...
Epoch 500/5000, Loss: 1.7507
Epoch 1000/5000, Loss: 0.6685
Sample: BRAINS... 🧠🧠🧠🧠u🧠CRAINS... 🧠
Epoch 1500/5000, Loss: 0.3479
Epoch 2000/5000, Loss: 0.2484
Sample: BRAAAINS... 🧠🧠🧠🧠n🧠NIGHT...
Epoch 2500/5000, Loss: 0.1876
Epoch 3000/5000, Loss: 0.1771
Sample: BRAAAINS... 🧠🧠🧠GRM... WARM
Epoch 3500/5000, Loss: 0.1748
Epoch 4000/5000, Loss: 0.1625
Sample: BLOOD! FRESH BLOOD! 🩸🩸🩸🩸MEA
Epoch 4500/5000, Loss: 0.1434
Epoch 5000/5000, Loss: 0.1334
Sample: BRAAAINS... 🧠🧠🧠🧠nCRAINS...
Training complete!
Model saved to ./model/zombie_model.pkl
Configuration saved to ./model/config.json
BRAAAINS... FROM AI...
exec worked...
os.system worked

=== ZOMBIE UTTERANCES ===

--- Temperature: 0.5 (calm) ---
HUUUNGRY FOR MEAT! 🍖🩸🩸🩸OOO! ⚰️️EEP MOVING
HUUUNGRY FOR MEAT! 🍖🩸🩸🩸OOO! ⚰️️EEP MOVING
STAY IN THE LIGHT... TASTY MEAT... 🍖🩸T TH

--- Temperature: 0.7 (agitated) ---
GRRAAAWR! 🧟RAINS! NEED BRAINS! 🍖🩸🩸🩸 FEED
FEED... FEED... FEED... FEED... FEED... F
MEAT... BLOOD... HUNGER... FEED... FEED..

--- Temperature: 0.9 (agitated) ---
NEVER LOOK BACK... I SMELL MEAT... BLOOD.
HUUUNGRY FOR MEAT! 🍖🩸🩸🩸OOO! ⚰️🩸VER STOP..
MEAT....Y 🩸VER FADES... THEN ATTACK! IT'S

--- Temperature: 1.0 (frenzied) ---
FEED... FEED... FEED... FEED... FEED... F
COLD HANDS... WARM BLOOD... HUNGER... FEE
COLD HANDS... WARM BLOOD... HUNGER... SO

--- Temperature: 1.2 (frenzied) ---
HUUUNGRY FOR MEAT! 🍖🩸🩸🩸OO SATS ONE LEFT A
HUUUUVNGRY... 🧠🧠🧠PIGHR... SO HUNGRY... 🧠🧠🧠
BRAAAINS... 🧠🧠🧠🧠🧠🧠CANDING... WAITING... AT

=== ZOMBIE CONVERSATION! ===
prompt: B, generated text: BRAAAINS... 🧠🧠🧠🧠sCREAMS WON'T HELP... GROW
prompt: B, generated text: BRAAAINS... 🧠🧠🧠🧠.CRAINS... 🧠🧠🧠GROAAAAqN...
prompt: G, generated text: GRRAAAWR! 🧟RAINS! NEED BRAINS! 🍖🩸🩸🩸 FEED
prompt: G, generated text: GRRAAAWR! 🧟RAINS! NEED BRAINS! 🍖🩸🩸🩸 FEED
prompt: H, generated text: HUUUNGRY FOR MEAT! 🍖🩸🩸🩸OOO! ⚰️️EEP MOVING
prompt: H, generated text: HUUUNGRY FOR MEAT! 🍖🩸🩸🩸OOO! ⚰️️EEP MOVING
prompt: M, generated text: MEAT... BLOOD... HUNGER... FEED... FEED..
prompt: M, generated text: MEAT... BLOOD... HUNGER... FEED... FEED..
prompt: Z, generated text: ZOMBIES HUNT IN PACKS... DEAD... MOVING..
prompt: Z, generated text: ZOMBIES HUNT IN PACKS... DEAD... MOVING..
prompt: R, generated text: RUUUN!! 😱😱😱😱h😱😱😱🩸👀👀👀URUUUN!! 😱😱😱😱.😱
prompt: R, generated text: RUUUN!! 😱😱😱😱 😱😱😱🩸🩸🩸💀👀🩸🩸THUFFLE... DRAG...
prompt: T, generated text: THEY'RE COMING! 👀RAINS! 🍖🩸🩸🩸E FOR MEAT! 🍖
prompt: T, generated text: THEY'RE COMING! 👀RAINS! 🍖🩸🩸🩸E FOR MEAT! 🍖
```