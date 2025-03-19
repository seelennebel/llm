from labels import id2label, label2id
from model_checkpoint import model_checkpoint

from datasets import load_from_disk
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer

dataset_path = "./dataset"
PII = load_from_disk(dataset_path)
if PII:
    print(f"Loaded dataset {dataset_path}")

tokenizer_path = "./preprocessing_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
if tokenizer:
    print(f"Loaded tokenizer {tokenizer_path}")

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=35, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="AM",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=PII["train"],
    eval_dataset=PII["validation"],
    processing_class=tokenizer
)

trainer.train()

model.save_pretrained("./AM")
tokenizer.save_pretrained("./AM_tokenizer")