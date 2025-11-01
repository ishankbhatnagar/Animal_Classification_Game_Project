from fastai.vision.all import *
import gradio as gr
from datasets import load_dataset
from pathlib import Path
import os

dataset = load_dataset("lucabaggi/animal-wildlife")

label_names = dataset["train"].features["label"].names

root = Path("./animal90_data")
(root/"train").mkdir(parents=True, exist_ok=True)
(root/"valid").mkdir(parents=True, exist_ok=True)

def save_split(split_name):
    for i, ex in enumerate(dataset[split_name]):
        label_id = ex["label"]
        label_name = label_names[label_id]
        img = PILImage.create(ex["image"])
        dest = root/split_name/label_name
        dest.mkdir(parents=True, exist_ok=True)
        img.save(dest/f"{i}.png")

if not any((root/"valid").rglob("*.png")):
    print("Saving dataset locally... (this will happen only once)")
    save_split("train")
    save_split("test")
    if (root/"test").exists():
        (root/"test").rename(root/"valid")
else:
    print("✅ Dataset already prepared.")

dls = ImageDataLoaders.from_folder(
    root, train='train', valid='valid',
    item_tfms=Resize(256),
    batch_tfms=aug_transforms(size=224, min_scale=0.75),
    bs=64
)

model_path = Path("animal90_classifier.pkl")

if model_path.exists():
    print("✅ Loading saved model...")
    learn = load_learner(model_path)
else:
    print("🚀 Training new model... (this will take ~1 hour on GPU)")
    learn = vision_learner(dls, resnet50, metrics=accuracy)
    learn.fine_tune(5)
    learn.export(model_path)
    print("💾 Model trained and saved as", model_path)

labels = learn.dls.vocab

def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    top3_idx = probs.argsort(descending=True)[:3]
    return {labels[i]: float(probs[i]) for i in top3_idx}

title = "🐾 Animal Species Classifier (90 Species)"
description = "A Fastai + ResNet50 model trained on lucabaggi/animal-wildlife dataset with 90 animal species."

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", label="Upload an Animal Image"),
    outputs=gr.Label(num_top_classes=3, label="Top Predictions"),
    title=title,
    description=description,
)

demo.launch(share=True)
