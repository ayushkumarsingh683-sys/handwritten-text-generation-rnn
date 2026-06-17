# handwritten-text-generation-rnn
AI Handwritten Text Generator ✒️ Built a character-level Recurrent Neural Network (RNN/LSTM) in PyTorch that mimics human hand movements. Instead of static images, it tracks sequential pen coordinates ($\Delta x, \Delta y$) and pen states to dynamically "draw" distinct cursive styles based on typed inputs like "cat".
# Text-Conditioned Handwritten Text Generation RNN

A character-level Recurrent Neural Network (RNN) implemented in PyTorch that simulates continuous human handwriting movements based on typed text input. 

Instead of generating static pixel images, this model tracks sequential pen-tip movements—learning the structural dynamics of letter formations to "draw" cursive-like strokes from scratch.

## 🚀 Features
- **Online Coordinate Processing:** Tracks relative changes ($\Delta x$, $\Delta y$) and binary pen states (up/down strokes).
- **Text-Conditioning Engine:** Uses a character embedding layer to map specific words to distinct drawing behaviors.
- **Probabilistic Generation:** Employs spatial distributions ($\mu, \sigma$) to sample organic, varied stroke outputs rather than rigid repetitions.

## 🛠️ Architecture Overview
The model passes embedded text context alongside historical pen stroke vectors into a deep **LSTM network**. The network translates these hidden states into probabilistic distribution parameters, allowing a bivariate normal sampling loop to iteratively construct the final canvas trajectory.



## 📦 Requirements & Setup
Ensure you have the core Python dependencies installed:
```bash
pip install torch numpy matplotlib
