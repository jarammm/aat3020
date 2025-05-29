import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from dataset import PersonaDataset
import torch
from utils import seed_everything


seed_everything(42)
model_name = "cosmoquester/bart-ko-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

df = pd.read_csv("train_dataset.csv", encoding='utf-8')

persona_list = {
    'informal': '구어체',
    'android': '안드로이드',
    'azae': '아재',
    'chat': '채팅',
    'choding': '초등학생',
    'emoticon': '이모티콘',
    'enfp': 'enfp',
    'gentle': '신사',
    'halbae': '할아버지',
    'halmae': '할머니',
    'joongding': '중학생',
    'king': '왕',
    'naruto': '나루토',
    'seonbi': '선비',
    'sosim': '소심한',
    'translator': '번역기'
}


special_tokens = [f"<{p}>" for p in persona_list.values()] + ['<formal>']
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model.resize_token_embeddings(len(tokenizer))


train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_dataset = PersonaDataset(train_df, tokenizer, persona_list)
val_dataset = PersonaDataset(val_df, tokenizer, persona_list)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

training_args = Seq2SeqTrainingArguments(
    output_dir="./bart-finetuned",
    per_device_train_batch_size=8,
    learning_rate=4e-5,
    num_train_epochs=3,
    logging_steps=1,
    save_steps=200,
    save_total_limit=5,
    eval_strategy="steps",
    eval_steps=200,
    report_to="wandb",
    fp16=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()