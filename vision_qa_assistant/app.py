import torch
from PIL import Image
import gradio as gr
from transformers import BlipProcessor, BlipForQuestionAnswering
import easyocr
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model & processor (first run downloads weights)
MODEL_NAME = "Salesforce/blip-vqa-base"
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def answer(image, audio, question):
    if image is None or not question or not question.strip():
        return "Please upload an image and type a question."
    try:
        qlow = question.lower()
        ocr_hint = ""
        if any(k in qlow for k in ["text", "read", "label", "sign", "price"]):
            ocr = ocr_reader.readtext(image, detail=0)
            if ocr:
                ocr_hint = " Detected text: " + " | ".join(ocr[:8])  # limit number of texts shown

        inputs = processor(images=image, text=question + ocr_hint, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=40)
        ans = processor.decode(output_ids[0], skip_special_tokens=True)
        return ans
    except Exception as e:
        return f"Error: {e}"

demo = gr.Interface(
    fn=answer,
    inputs=[
    gr.Image(type="pil", label="Upload an image"),
    gr.Audio(sources=["microphone"], type="filepath", label="Ask by voice (or use text below)"),
    gr.Textbox(label="Or type your question", placeholder="e.g., What fruit is this?", lines=1)
],
    outputs=gr.Textbox(label="Answer"),
    title="Vision Q&A Assistant (BLIP VQA)",
    description="Upload an image and ask a question. The model will answer based on visual content."
)

if __name__ == "__main__":
    demo.launch()
sample_images = ["samples/cat.jpg", "samples/plant.jpg", "samples/car.jpg"]
sample_questions = [
    "What animal is this?",
    "What plant is this?",
    "What color is the car?"
]

demo = gr.Interface(
    fn=answer,
    inputs=[
        gr.Image(type="pil", label="Upload an image"),
        gr.Textbox(label="Ask a question about the image", placeholder="e.g., What fruit is this?")
    ],
    outputs=gr.Textbox(label="Answer"),
    examples=[[img, q] for img, q in zip(sample_images, sample_questions)],
    title="Vision Q&A Assistant (BLIP VQA)",
    description="Upload an image and ask a question. The model will answer based on visual content."
)
import gradio as gr

def audio_qa(image, audio):
    question = transcribe(audio)  # Convert speech → text
    return predict(image, question)

iface = gr.Interface(
    fn=audio_qa,
    inputs=[gr.Image(type="filepath"), gr.Audio(type="filepath")],
    outputs="text",
    title="Vision Q&A Assistant (with Audio)",
    description="Upload an image and ask a question by voice. The model will answer based on visual content."
)

iface.launch()
import whisper


whisper_model = whisper.load_model("base")

def transcribe(audio_file):
   
    result = whisper_model.transcribe(audio_file)
    return result["text"]
