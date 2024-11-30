import mlflow
import mlflow.pytorch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
import os
from fastapi import FastAPI
import torch

# Désactiver wandb
os.environ["WANDB_DISABLED"] = "true"

# Charger les données d'entraînement et créer une colonne 'text'
data = pd.read_csv("../data/lol_data.csv")
data['text'] = data['Question'] + " " + data['Answer']

# Diviser les données en ensembles d'entraînement et de validation
train_df, val_df = train_test_split(data[['text']], test_size=0.2, random_state=42)

# Convertir en Dataset et supprimer les colonnes inutiles
train_dataset = Dataset.from_pandas(train_df).remove_columns(["__index_level_0__"])
eval_dataset = Dataset.from_pandas(val_df).remove_columns(["__index_level_0__"])

# Charger le modèle et le tokenizer, et définir un token de remplissage
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Tokeniser le dataset avec les labels
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=50)
    tokens["labels"] = tokens["input_ids"].copy()  # Utiliser input_ids comme labels
    return tokens

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    remove_unused_columns=False  # Garde les colonnes nécessaires pour le calcul de la perte
)

# Configurer MLflow pour pointer vers votre serveur local
mlflow.set_tracking_uri("../mlruns")
mlflow.set_experiment("Chatbot_Training")

# Initialiser le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Entraîner le modèle et enregistrer avec MLflow
with mlflow.start_run() as run:
    trainer.train()

    # Log des paramètres et du modèle dans MLflow
    mlflow.log_params({"model_name": model_name, "epochs": training_args.num_train_epochs})
    #mlflow.pytorch.log_model(model, artifact_path="model", registered_model_name="OurModel")
    mlflow.pytorch.log_model(model, "model")

# FastAPI

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.post("/predict")
async def answer_prompt(prompt: str):
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
    answer = tokenizer.batch_decode(generated_ids)[0]
    return {"answer": answer}