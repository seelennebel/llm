from transformers import DistilBertForTokenClassification

model = DistilBertForTokenClassification.from_pretrained("./AM_modified")

model.push_to_hub("AM")