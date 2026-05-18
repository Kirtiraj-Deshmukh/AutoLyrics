import torch
from transformers import WhisperForConditionalGeneration
from peft import PeftModel
from jiwer import wer, cer

def evaluate_predictions(base_model_name, adapter_path, eval_data, processor, device):
    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name).to(device)

    peft_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    peft_model = PeftModel.from_pretrained(peft_model, adapter_path).to(device)

    references = []
    base_preds = []
    lora_preds = []

    for sample in eval_data.select(range(min(10, len(eval_data)))):
        input_feats = torch.tensor([sample["input_features"]]).to(device)

        ref_ids = [i for i in sample["labels"] if i != -100]
        ref_text = processor.tokenizer.decode(ref_ids, skip_special_tokens=True)
        references.append(ref_text)

        with torch.no_grad():
            base_gen = base_model.generate(input_features=input_feats)
            base_pred = processor.tokenizer.decode(base_gen[0], skip_special_tokens=True)
            base_preds.append(base_pred)

            lora_gen = peft_model.generate(input_features=input_feats)
            lora_pred = processor.tokenizer.decode(lora_gen[0], skip_special_tokens=True)
            lora_preds.append(lora_pred)

    base_wer = wer(references, base_preds)
    lora_wer = wer(references, lora_preds)
    base_cer = cer(references, base_preds)
    lora_cer = cer(references, lora_preds)

    print("Results:")
    print(f"Base Model ({base_model_name}):- WER: {base_wer:.4f} % | CER: {base_cer:.4f} %")
    print(f"LoRA Fine-Tuned Model:-   WER: {lora_wer:.4f} % | CER: {lora_cer:.4f} %")
    
    if base_wer > 0:
        improvement = ((base_wer - lora_wer) / base_wer) * 100
        print(f"Relative WER Reduction: {improvement:.2f}%")