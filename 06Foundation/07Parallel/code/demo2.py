# !/usr/bin/python
# -*- coding: utf-8 -*-
 
import nltk
import torch
import evaluate
import datasets
import numpy as np
from nltk.tokenize import sent_tokenize
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
 
nltk.download("punkt")
 
dataset_name = "samsum" # 数据集名称
model_name="谷歌/flan-t5-xxl" # 模型名称
max_input_length = 512
max_gen_length = 128
output_dir = "checkpoints"
num_train_epochs = 5
learning_rate = 5e-5
deepspeed_config = "./ds_config.json" # deepspeed 配置文件
per_device_train_batch_size=1 # batch size 设置为 1，因为太大导致 OOM
per_device_eval_batch_size=1
gradient_accumulation_steps=2 # 由于单卡的 batch size 为 1，为了扩展 batch size，使用梯度累加
 
tokenizer = AutoTokenizer.from_pretrained(model_name)
 
# 加载数据
dataset = datasets.load_dataset(dataset_name)
print(dataset["train"][0])
 
# tokenize
def preprocess(examples):
    dialogues = ["summarize:" + dia for dia in examples["dialogue"]]
    # summaries = [summ for summ in examples["summary"]]
    model_inputs = tokenizer(dialogues, max_length=max_input_length, truncation=True)
    labels = tokenizer(text_target=examples["summary"], max_length=max_gen_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
 
tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["dialogue", "summary", "id"])
# print(tokenized_dataset["train"]["input_ids"][0]) # 打印结果
 
 
# 对 batch 进行 padding
def collate_fn(features):
    batch_input_ids = [torch.LongTensor(feature["input_ids"]) for feature in features]
    batch_attention_mask = [torch.LongTensor(feature["attention_mask"]) for feature in features]
    batch_labels = [torch.LongTensor(feature["labels"]) for feature in features]
 
    batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
    batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100)
 
    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels
    }
# 用于测试的代码
# dataloader = DataLoader(tokenized_dataset["test"], shuffle=False, batch_size=4, collate_fn=collate_fn)
# batch = next(iter(dataloader))
# print(batch)
 
 
# 加载模型
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# 用于测试的代码
# dataloader = DataLoader(tokenized_dataset["test"], shuffle=False, batch_size=4, collate_fn=collate_fn)
# batch = next(iter(dataloader))
# output = model(**batch)
# print(output)
 
 
# 定义评估函数
metric = evaluate.load("rouge")
 
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result
 
 
# 设置训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    eval_accumulation_steps=1, # 防止评估时导致 OOM
    predict_with_generate=True,
    fp16=False,
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
    # logging & evaluation strategies
    logging_dir="logs",
    logging_strategy="steps",
    logging_steps=50, # 每 50 个 step 打印一次 log
    evaluation_strategy="steps",
    eval_steps=500, # 每 500 个 step 进行一次评估
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    deepspeed=deepspeed_config, # deepspeed 配置文件的位置
    report_to="all"
)
 
 
# 模型训练
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)
 
trainer.train()
# 打印验证集上的结果
print(trainer.evaluate(tokenized_dataset["validation"]))
# 打印测试集上的结果
print(trainer.evaluate(tokenized_dataset["test"]))
# 保存最优模型
trainer.save_model("best")