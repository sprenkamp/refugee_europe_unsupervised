from transformers import pipeline
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-zle-en")
print(pipe("Скільки мені слід купити пива?"))


