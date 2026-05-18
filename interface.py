import gradio as gr
from transformers import pipeline, WhisperForConditionalGeneration
from peft import PeftModel

def launch_interface(model_name, adapter_path, processor, device):
    base_model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None
    )
    base_model.eval()
    base_model.config.use_cache = True

    peft_model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None
    )
    peft_model = PeftModel.from_pretrained(peft_model, adapter_path)
    peft_model.eval()
    peft_model.config.use_cache = True

    pipe_device = 0 if device == "cuda" else -1

    tuned_pipe = pipeline(
        "automatic-speech-recognition",
        model=peft_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        device=pipe_device
    )

    base_pipe = pipeline(
        "automatic-speech-recognition",
        model=base_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        device=pipe_device
    )

    def transcribe_long_audio(audio_file_path):
        if audio_file_path is None:
            return "No audio provided.", "No audio provided."

        tuned_result = tuned_pipe(
            audio_file_path,
            generate_kwargs={"max_new_tokens": 225, "num_beams": 3}
        )

        base_result = base_pipe(
            audio_file_path,
            generate_kwargs={"max_new_tokens": 225, "num_beams": 3}
        )

        return tuned_result["text"], base_result["text"]

    demo = gr.Interface(
        fn=transcribe_long_audio,
        inputs=gr.Audio(type="filepath", label="Upload Full Song"),
        outputs=[
            gr.Textbox(label="Fine-Tuned AutoLyrics Output", lines=10, max_lines=20),
            gr.Textbox(label="Baseline Model Output", lines=10, max_lines=20)
        ],
        title="AutoLyrics"
    )

    demo.launch(share=True)