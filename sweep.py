import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from dataset import PersonaDataset
import torch
import wandb

model_name = "cosmoquester/bart-ko-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

df = pd.read_csv("cleaned_smilestyle_dataset.tsv", sep="\t")


persona_list = ['android', 'azae', 'chat', 'choding', 'emoticon', 'enfp',
                'gentle', 'halbae', 'halmae', 'joongding', 'king', 'naruto',
                'seonbi', 'sosim', 'translator']


special_tokens = [f"<{p}>" for p in persona_list]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model.resize_token_embeddings(len(tokenizer))


train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = PersonaDataset(train_df.reset_index(drop=True), tokenizer, persona_list)
val_dataset = PersonaDataset(val_df.reset_index(drop=True), tokenizer, persona_list)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


wandb.init()
config = wandb.config

training_args = Seq2SeqTrainingArguments(
    output_dir="./sweep-output",
    per_device_train_batch_size=config.per_device_train_batch_size,
    learning_rate=config.learning_rate,
    num_train_epochs=config.num_train_epochs,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=1,
    report_to="wandb",
    fp16=torch.cuda.is_available(),
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