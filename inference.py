from transformers import pipeline

qa_model = pipeline(task="question-answering", model="./models/qa-bert-base", tokenizer="./models/qa-bert-base")
question = "Where is the github link?"
context = "You can find the github link for this video in the description."
print(qa_model(question=question, context=context))
