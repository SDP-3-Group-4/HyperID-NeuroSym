"""
demo_toy_training_walkthrough.py
----------------------------------------------------------------------
LEARNING TOOL ONLY. This is NOT part of the real pipeline and does not
replace train_trait_model.py. It uses tiny made-up numbers instead of
real CLIP embeddings, so you can run it instantly and watch every step
with your own eyes.

Run it with: python demo_toy_training_walkthrough.py
(only needs `pip install torch`, nothing else)
----------------------------------------------------------------------
"""

import torch
import torch.nn as nn

print("=" * 70)
print("STEP 0: Here are our fake 'photos' -- 3 of them.")
print("In reality these numbers come from running a real photo through")
print("CLIP. Here we just type in small made-up numbers so everything")
print("fits on screen. Real CLIP embeddings have 512 numbers each; ours")
print("have 4, purely for visibility.")
print("=" * 70)

# Each row = one fake photo. We're pretending:
#   photo 1 = a Beagle       photo 2 = another Beagle      photo 3 = a Pug
fake_embeddings = torch.tensor([
    [0.9, 0.1, 0.4, 0.2],
    [0.8, 0.2, 0.5, 0.1],
    [0.1, 0.9, 0.2, 0.8],
])
photo_breeds = ["Beagle", "Beagle", "Pug"]

print(f"\nFake embeddings:\n{fake_embeddings}")
print(f"We're told (from the dataset) these photos are of: {photo_breeds}\n")


print("=" * 70)
print("STEP 1: WEAK SUPERVISION -- this is the step you were asking about.")
print("We do NOT hand-label each photo's traits ourselves. We look up")
print("each photo's breed in the ontology, and copy that breed's known")
print("trait value -- turning a breed label into a trait label, for free.")
print("=" * 70)

# Stands in for what our SPARQL query pulls from the real ontology file
# (load_breed_traits_from_ontology() in the real script). Only ONE trait
# dimension (EarShape) is used here to keep it simple -- the real script
# does this for all six trait dimensions at once.
ontology_ear_shape = {"Beagle": "FloppyEars", "Pug": "ButtonEars"}

photo_trait_labels = [ontology_ear_shape[breed] for breed in photo_breeds]
print(f"Weakly-supervised labels (auto-generated, not hand-labeled): {photo_trait_labels}\n")

label_to_index = {"FloppyEars": 0, "ButtonEars": 1}
index_to_label = {0: "FloppyEars", 1: "ButtonEars"}
y = torch.tensor([label_to_index[label] for label in photo_trait_labels])
print(f"Same labels, as numbers the model can use: {y.tolist()}\n")


print("=" * 70)
print("STEP 2: Build the tiniest possible trait-predicting model.")
print("Input: 4 numbers (our fake embedding). Output: 2 scores, one per")
print("possible EarShape value. Untrained -- starts out random/useless.")
print("=" * 70)

torch.manual_seed(0)
model = nn.Linear(4, 2)   # real version: bigger network, 6 heads, one per trait dimension
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

with torch.no_grad():
    starting_scores = model(fake_embeddings)
print(f"Untrained scores (garbage, hasn't learned anything yet):\n{starting_scores}\n")


print("=" * 70)
print("STEP 3: TRAINING LOOP -- watch the loss go down.")
print("Each step: model guesses -> compare guess to the weak label ->")
print("loss = how wrong it was -> nudge the model's numbers to be less wrong.")
print("=" * 70)

for step in range(10):
    optimizer.zero_grad()
    predictions = model(fake_embeddings)   # model guesses
    loss = loss_fn(predictions, y)          # compare guess to weak label
    loss.backward()                          # figure out which direction to adjust
    optimizer.step()                          # actually adjust the model's numbers
    print(f"  step {step + 1}: loss = {loss.item():.4f}")
print()


print("=" * 70)
print("STEP 4: Use the (now trained) model on a brand NEW fake photo.")
print("=" * 70)

new_fake_photo = torch.tensor([[0.85, 0.15, 0.45, 0.15]])  # looks similar to our Beagle photos
with torch.no_grad():
    final_scores = model(new_fake_photo)
    predicted_index = final_scores.argmax(dim=1).item()

print(f"New photo's fake embedding: {new_fake_photo.tolist()}")
print(f"Model's predicted EarShape: {index_to_label[predicted_index]}")
print()
print("That prediction -- a plain word, 'FloppyEars' -- is exactly what")
print("gets handed to the SPARQL query in the real pipeline. This is the")
print("moment perception (numbers) becomes a symbol the ontology understands.")
