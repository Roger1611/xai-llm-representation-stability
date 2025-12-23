import torch
import matplotlib.pyplot as plt
import umap
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# 1. Load pretrained model
# -----------------------------
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name,
    output_hidden_states=True   # VERY IMPORTANT for XAI
)

model.eval()  # inference mode (no training)

# -----------------------------
# 2. Sentences with SAME word, DIFFERENT meanings
# -----------------------------
finance_sentences = [
    "I deposited money in the bank",
    "She withdrew cash from the bank",
    "The bank approved the loan",
    "He opened a savings account at the bank",
    "The bank offered a low interest rate"
]

river_sentences = [
    "The river bank was muddy",
    "Children played near the bank of the river",
    "The boat reached the bank",
    "They sat quietly on the river bank",
    "Trees grew along the bank of the river"
]

sentences = finance_sentences + river_sentences
labels = ["finance"] * len(finance_sentences) + ["river"] * len(river_sentences)

# -----------------------------
# 3. Choose layer to analyze
# -----------------------------
# Early layer: 2
# Middle layer: 4
# Late layer: -1 (last)
LAYER_ID = -1

# -----------------------------
# 4. Extract hidden states of the word "bank"
# -----------------------------
vectors = []

with torch.no_grad():
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt")
        outputs = model(**inputs)

        hidden_states = outputs.hidden_states
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Find index of token "bank"
        if "bank" not in tokens:
            continue

        bank_index = tokens.index("bank")

        # Extract hidden state vector
        vec = hidden_states[LAYER_ID][0, bank_index, :].cpu().numpy()
        vectors.append(vec)

# -----------------------------
# 5. Reduce dimension with UMAP
# -----------------------------
reducer = umap.UMAP(
    n_neighbors=10,
    min_dist=0.3,
    random_state=42
)

embeddings_2d = reducer.fit_transform(vectors)

# -----------------------------
# 6. Plot results
# -----------------------------
plt.figure(figsize=(7, 6))

for i, label in enumerate(labels):
    if label == "finance":
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                    color="blue", label="Finance" if i == 0 else "")
    else:
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                    color="red", label="River" if i == len(finance_sentences) else "")

plt.title("UMAP of Hidden State Representations for token 'bank'")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend()
plt.grid(True)

# Save for paper
plt.savefig("experiment1_bank_representation.png", dpi=300, bbox_inches="tight")
plt.show()
