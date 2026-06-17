import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- 1. VOCABULARY MAP (Text to Numbers) ---
# Added 'c' and 't' to the dictionary to support the word "cat"
char_to_idx = {' ': 0, 'a': 1, 'b': 2, 'c': 3, 'u': 4, 'n': 5, 't': 6}
vocab_size = len(char_to_idx)

# --- 2. GENERATE TEXT-CONDITIONED SYNTHETIC DATA ---
def generate_conditioned_data(num_sequences=120, seq_len=60):
    stroke_data = []
    text_data = []
    
    for i in range(num_sequences):
        # FIXED: Added padding spaces so all sequences are exactly 3 characters long
        if i % 3 == 0:
            word = "un "
        elif i % 3 == 1:
            word = "an "
        else:
            word = "cat"
            
        text_tokens = [char_to_idx[c] for c in word]
        text_data.append(text_tokens)
        
        seq = []
        x_prev, y_prev = 0.0, 0.0
        for t in range(seq_len):
            x_curr = t * 0.2
            
            # Map unique structural drawing trajectories to each word
            if word == "un ":
                y_curr = np.sin(x_curr)                         # Smooth continuous curve
            elif word == "an ":
                y_curr = np.abs(np.sin(x_curr))                # Sharp bouncing bumps
            elif word == "cat":
                y_curr = np.sin(x_curr) * np.cos(x_curr * 1.5)  # Unique loops for "cat"
                
            dx = x_curr - x_prev
            dy = y_curr - y_prev
            pen_up = 1.0 if t == seq_len - 1 else 0.0
            
            seq.append([dx, dy, pen_up])
            x_prev, y_curr = x_curr, y_curr
            
        stroke_data.append(seq)
        
    return torch.tensor(stroke_data, dtype=torch.float32), torch.tensor(text_data, dtype=torch.long)

# --- 3. CLEANED MODEL DEFINITION ---
class TextConditionedHandwritingRNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64):
        super(TextConditionedHandwritingRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.char_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Takes stroke vector (3) + text context vector (hidden_dim)
        self.lstm = nn.LSTM(input_size=3 + hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim + hidden_dim, 5)

    def forward(self, stroke_input, text_input, hidden=None, prev_window=None):
        batch_size = stroke_input.size(0)
        
        # Embed the text tokens into continuous vectors
        embedded_text = self.char_embedding(text_input) 
        
        # Average the character embeddings to create a static context window
        current_window = embedded_text.mean(dim=1, keepdim=True)

        # Combine our stroke parameters with the text guidance vector
        lstm_input = torch.cat([stroke_input, current_window.expand(batch_size, stroke_input.size(1), self.hidden_dim)], dim=-1)
        lstm_out, hidden = self.lstm(lstm_input, hidden)
        
        # Combine LSTM output with our text window for final predictions
        combined_output = torch.cat([lstm_out, current_window.expand(batch_size, stroke_input.size(1), self.hidden_dim)], dim=-1)
        pred = self.fc(combined_output)
        
        mu_x = pred[:, :, 0]
        mu_y = pred[:, :, 1]
        sigma_x = torch.exp(pred[:, :, 2])
        sigma_y = torch.exp(pred[:, :, 3])
        pen_status = torch.sigmoid(pred[:, :, 4])
        
        return mu_x, mu_y, sigma_x, sigma_y, pen_status, hidden

# --- 4. LOSS FUNCTION ---
def compute_loss(mu_x, mu_y, sigma_x, sigma_y, pen_status, targets):
    target_dx, target_dy, target_pen = targets[:, :, 0], targets[:, :, 1], targets[:, :, 2]
    loss_x = torch.log(sigma_x) + 0.5 * ((target_dx - mu_x) / sigma_x)**2
    loss_y = torch.log(sigma_y) + 0.5 * ((target_dy - mu_y) / sigma_y)**2
    return torch.mean(loss_x + loss_y) + F.binary_cross_entropy(pen_status, target_pen)

# --- 5. INITIALIZE AND TRAIN ---
stroke_dataset, text_dataset = generate_conditioned_data()
model = TextConditionedHandwritingRNN(vocab_size=vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training text-conditioned model (un, an, cat)...")
model.train()
for epoch in range(151):
    optimizer.zero_grad()
    
    inputs = stroke_dataset[:, :-1, :]
    targets = stroke_dataset[:, 1:, :]
    
    mu_x, mu_y, sigma_x, sigma_y, pen_status, _ = model(inputs, text_dataset)
    loss = compute_loss(mu_x, mu_y, sigma_x, sigma_y, pen_status, targets)
    
    loss.backward()
    optimizer.step()
    if epoch % 30 == 0:
        print(f"Epoch {epoch}/150 | Loss: {loss.item():.4f}")

# --- 6. GENERATION BASED ON A TYPED WORD PROMPT ---
model.eval()

# Setting prompt to write out our brand new word: "cat"
prompt_word = "cat" 
print(f"\nGenerating handwriting path specifically conditioned on the text: '{prompt_word}'")

prompt_tokens = torch.tensor([[char_to_idx[c] for c in prompt_word]], dtype=torch.long)
generated_strokes = [[0.0, 0.0, 0.0]]
hidden = None

with torch.no_grad():
    curr_input = torch.tensor([[generated_strokes[-1]]], dtype=torch.float32)
    for _ in range(60):
        mu_x, mu_y, sigma_x, sigma_y, pen_status, hidden = model(curr_input, prompt_tokens, hidden)
        
        next_dx = torch.normal(mu_x, sigma_x).item()
        next_dy = torch.normal(mu_y, sigma_y).item()
        next_pen = 1.0 if pen_status.item() > 0.5 else 0.0
        
        generated_strokes.append([next_dx, next_dy, next_pen])
        curr_input = torch.tensor([[[next_dx, next_dy, next_pen]]], dtype=torch.float32)

# --- 7. DISPLAY THE TEXT-MATCHED RESULTS ---
strokes = np.array(generated_strokes)
x_coords = np.cumsum(strokes[:, 0])
y_coords = np.cumsum(strokes[:, 1])

plt.figure(figsize=(8, 3))
plt.plot(x_coords, y_coords, color='teal', marker='o', markersize=3, label=f"Result for '{prompt_word}'")
plt.title("Text-Conditioned Generation Result")
plt.legend()
plt.show()