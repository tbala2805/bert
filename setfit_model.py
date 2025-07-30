# setfit_model.py
from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset

# Load data
train_data = [
    {"text": "I love my dog", "label": 0},
    {"text": "This is a great movie", "label": 1}
]
train_dataset = Dataset.from_list(train_data)

# Load a pre-trained sentence transformer
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

# Train the model
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    loss_class="softmax_loss",
    metric="accuracy"
)
trainer.train()

# Save the model
model.save_pretrained("./setfit_model")
