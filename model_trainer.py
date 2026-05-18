import torch
from transformers import (
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model
from jiwer import wer

def compute_metrics_fn(processor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer_score = 100 * wer(label_str, pred_str)
        return {"wer": wer_score}
    return compute_metrics

def train_model(model_name, train_dataset, eval_dataset, processor, data_collator, device, out_DIR, size_batch, grad_steps, rate_learning, max_steps):
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None
    )
    model.config.use_cache = False
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.enable_input_require_grads()
    
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = train_dataset.select_columns(["input_features", "labels"])
    eval_dataset = eval_dataset.select_columns(["input_features", "labels"])

    training_args = Seq2SeqTrainingArguments(
        output_dir=out_DIR,
        per_device_train_batch_size=size_batch,
        gradient_accumulation_steps=grad_steps,
        learning_rate=rate_learning,
        warmup_steps=10,
        max_steps=max_steps,
        gradient_checkpointing=True,
        fp16=True if device == "cuda" else False,
        eval_strategy="steps",
        per_device_eval_batch_size=size_batch,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=50,
        eval_steps=50,
        logging_steps=10,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        label_names=["labels"],
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn(processor),
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(out_DIR)
    processor.save_pretrained(out_DIR)