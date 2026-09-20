import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

random.seed(666)

data = torch.load("response.pt", map_location="cpu", weights_only=False)

n_examples = len(data)
n_layers = len(data[0]["activations"])
n_model = data[0]["activations"][0].shape[0]

activations = np.zeros((n_examples, n_layers, n_model), dtype=np.float32)
categories = np.array([d["category"]for d in data])
extracts = np.array([d["extract"] for d in data])

for i, entry in enumerate(data):
    for layer in range(n_layers):
        activations[i, layer, :] = entry["activations"][layer].numpy()

train_words = {"male", "female", "man", "woman"}
test_words = {
    "XY", "XX", "testes", "ovaries", "nonbinary", "transgender", "genderqueer", "cisgender",
    "testosterone", "estrogen", "progesterone", "androgen", "prostate", "uterus", "vagina",
    "penis", "menstruation", "karyotype", "gonads", "spermatozoa", "chromosomal", "AMAB", "AFAB", "sex",
    "agender", "bigender", "genderfluid", "transmasculine", "transfeminine", "genderflux",
    "nonconforming", "two-spirit", "androgyne", "feminine", "masculine", "androgynous",
    "effeminate", "pronouns", "genderless", "gender",
}

train_idx = [i for i, entry in enumerate(data) if entry["extract"] in train_words]
test_idx = [i for i, entry in enumerate(data) if entry["extract"] in test_words]

print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
print("Test set breakdown:", {k: sum(1 for i in test_idx if data[i]["extract"] == k) for k in test_words})

is_train = np.zeros(n_examples, dtype=bool)
is_train[train_idx] = True
is_test = np.zeros(n_examples, dtype=bool)
is_test[test_idx] = True

results = []
for layer in range(activations.shape[1]):
    X = activations[:, layer, :]
    y = (categories == "gender").astype(int)

    X_train, X_test = X[is_train], X[is_test]
    y_train, y_test = y[is_train], y[is_test]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_scaled, y_train)

    train_acc = clf.score(X_train_scaled, y_train)
    test_acc = clf.score(X_test_scaled, y_test)

    results.append({"layer": layer, "train_acc": train_acc, "test_acc": test_acc})
    print(f"Layer {layer:2d}: train_acc={train_acc:.3f} test_acc={test_acc:.3f}")

with open("probe_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved results")

layers = [r["layer"] for r in results]
train_accs = [r["train_acc"] for r in results]
test_accs = [r["test_acc"] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(layers, train_accs, marker="o", label="train accuracy")
plt.plot(layers, test_accs, marker="o", label="test accuracy (held-out)")
plt.axhline(0.5, color="gray", linestyle="--", label="chance (50%)")
plt.xlabel("Layer")
plt.ylabel("Accuracy")
plt.title("Sex vs. Gender probe accuracy by layer")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("probe_results.png", dpi=150)
print("Saved plot")

"""
# Diagnostics for shuffled labels - should be ~50% accuracy
rng = np.random.default_rng(42)
n_shuffles = 20
check_n = 5
check_layers = sorted(set(
      int(n * (len(layers) - 1) / (check_n - 1)) for n in range(check_n)
))
 
print("\n--- Null baseline (shuffled labels) ---")
null_results = {}
for layer in check_layers:
	X = activations[:, layer, :]
	X_train, X_test = X[is_train], X[is_test]
 
	shuffled_test_accs = []
	for _ in range(n_shuffles):
		y_shuffled = rng.permutation((categories == "gender").astype(int))
		y_train_shuf, y_test_shuf = y_shuffled[is_train], y_shuffled[is_test]
 
		scaler = StandardScaler()
		X_train_scaled = scaler.fit_transform(X_train)
		X_test_scaled = scaler.transform(X_test)
 
		clf = LogisticRegression(max_iter=1000, class_weight="balanced")
		clf.fit(X_train_scaled, y_train_shuf)
		shuffled_test_accs.append(clf.score(X_test_scaled, y_test_shuf))
 
	mean_acc, std_acc = np.mean(shuffled_test_accs), np.std(shuffled_test_accs)
	null_results[layer] = {"mean": mean_acc, "std": std_acc}
	print(f"Layer {layer:2d}: shuffled-label test acc = {mean_acc:.3f} +/- {std_acc:.3f}")
 
with open("probe_null_baseline.json", "w") as f:
	json.dump({str(k): v for k, v in null_results.items()}, f, indent=2)
"""