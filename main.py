import config
from data_processing import load_and_prepare_dataset, DataCollatorSpeechSeq2SeqWithPadding
from training import train_model
from evaluation import evaluate_predictions
from interface import launch_interface

def main():
    #Loading and preprocessing dataset
    train_dataset, eval_dataset, processor = load_and_prepare_dataset(
        config.model_name, config.dataset_name
    )
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    #Training pipeline
    train_model(
        model_name=config.model_name,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processor=processor,
        data_collator=data_collator,
        device=config.device,
        out_DIR=config.out_DIR,
        size_batch=config.size_batch,
        grad_steps=config.grad_steps,
        rate_learning=config.rate_learning,
        max_steps=config.max_steps
    )

    #Evaluation
    evaluate_predictions(
        base_model_name=config.model_name,
        adapter_path=config.out_DIR,
        eval_data=eval_dataset,
        processor=processor,
        device=config.device
    )

    #Gradio
    launch_interface(
        model_name=config.model_name,
        adapter_path=config.out_DIR,
        processor=processor,
        device=config.device
    )

if __name__ == "__main__":
    main()