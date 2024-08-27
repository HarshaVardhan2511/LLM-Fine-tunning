pip install transformers torch datasets
from datasets import load_dataset
dataset = load_dataset("imdb")
print(dataset)
from datasets import load_dataset
from transformers import BertTokenizer
# Load the IMDb dataset
dataset = load_dataset("imdb")
print(dataset)
# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Define the tokenization function
def tokenize_function(examples):
 return tokenizer(examples['text'], padding="max_length", truncation=True)
# Apply the tokenization function to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
print(tokenized_datasets)
train_testvalid = tokenized_datasets['train'].train_test_split(test_size=0.01)
train_dataset = train_testvalid['train']
valid_dataset = train_testvalid['test']
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)
valid_dataloader = DataLoader(valid_dataset, batch_size=1)
from transformers import BertForSequenceClassification, AdamW
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
output_dir='./results',
evaluation_strategy="epoch",
learning_rate=2e-5,
per_device_train_batch_size=1,
per_device_eval_batch_size=1,
num_train_epochs=0.001,
weight_decay=0.01,
)
# Define Trainer with model, arguments, and datasets
trainer = Trainer(
model=model,
args=training_args,
train_dataset=train_dataset,
eval_dataset=valid_dataset
)
# Start training
trainer.train()
metrics = trainer.evaluate()
print(metrics)
predictions = trainer.predict(valid_dataset)
print(predictions)