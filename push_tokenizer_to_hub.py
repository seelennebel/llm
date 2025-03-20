from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("./AM_tokenizer")

tokenizer.push_to_hub("AM_tokenizer", token="")