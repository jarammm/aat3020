import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import wandb
import torch
from dataset import PersonaDataset
from utils import seed_everything


persona_list = [
    'android', 'azae', 'chat', 'choding', 'emoticon', 'enfp',
    'gentle', 'halbae', 'halmae', 'joongding', 'king', 'naruto',
    'seonbi', 'sosim', 'translator'
]


def train():
    wandb.init()
    config = wandb.config
    seed_everything(42)
    
    global tokenizer

    # 모델 및 토크나이저 로드
    model_name = "cosmoquester/bart-ko-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # 스페셜 토큰 등록
    special_tokens = [f"<{p}>" for p in persona_list]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))

    # 데이터 로드
    df = pd.read_csv("cleaned_smilestyle_dataset.tsv", sep="\t")
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    n_train = 900
    n_val = 100

    train_df = train_df[:n_train]
    val_df = val_df[:n_val]

    train_dataset = PersonaDataset(train_df, tokenizer, persona_list)
    val_dataset = PersonaDataset(val_df, tokenizer, persona_list)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./sweep-output",
        per_device_train_batch_size=config["per_device_train_batch_size"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],

        # Logging & Eval & Save
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        report_to="wandb",
        fp16=False)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    trainer.train()
    
if __name__ == '__main__':
  train()