from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
import torch
import sys
from labels import id2label

def output_am(text):
    tokenizer = AutoTokenizer.from_pretrained("./AM_tokenizer")    
    model = AutoModelForTokenClassification.from_pretrained("./AM")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_classes = torch.argmax(probabilities, dim=-1)
    for value in predicted_classes[0]:
        print(id2label[int(value)])

if __name__ == "__main__":
    output_am(sys.argv)