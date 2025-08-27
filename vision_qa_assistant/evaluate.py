import csv, torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

MODEL_NAME = "Salesforce/blip-vqa-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
model.eval()

def predict(img_path, question):
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, text=question, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_length=32)
    return processor.decode(out_ids[0], skip_special_tokens=True).strip().lower()

gold, correct = 0, 0
with open("eval.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pred = predict(row["image_path"], row["question"])
        ans  = row["answer"].strip().lower()
        gold += 1
        correct += int(pred == ans)
        print(f"Q: {row['question']} | pred: {pred} | gold: {ans}")

print(f"\nExact-match accuracy: {correct}/{gold} = {correct/gold:.2%}")
