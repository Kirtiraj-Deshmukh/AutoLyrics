import torch
from datasets import load_dataset, Audio
from transformers import WhisperProcessor
from dataclasses import dataclass
from typing import Any, Dict, List, Union

def load_and_prepare_dataset(model_name, dataset_name):
    raw_dataset = load_dataset(dataset_name)
    raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=16000))

    processor = WhisperProcessor.from_pretrained(model_name, language="english", task="transcribe")

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        target_text = batch.get("text", batch.get("lyrics", ""))
        batch["labels"] = processor.tokenizer(target_text).input_ids
        return batch

    train_split = raw_dataset["train"].select(range(min(25, len(raw_dataset["train"]))))
    test_split = raw_dataset["test"].select(range(min(10, len(raw_dataset["test"])))) if "test" in raw_dataset else train_split

    train_dataset = train_split.map(prepare_dataset, remove_columns=train_split.column_names)
    eval_dataset = test_split.map(prepare_dataset, remove_columns=test_split.column_names)
    
    return train_dataset, eval_dataset, processor

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch