# Zombie Model

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