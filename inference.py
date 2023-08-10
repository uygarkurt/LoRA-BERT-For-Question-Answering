from peft import PeftConfig, PeftModel
from transformers import BertForQuestionAnswering, BertTokenizerFast, pipeline


def lora_infer(question, context, MODEL_PATH):
    config = PeftConfig.from_pretrained(MODEL_PATH)
    model = PeftModel.from_pretrained(BertForQuestionAnswering.from_pretrained(config.base_model_name_or_path), MODEL_PATH)
    tokenizer = BertTokenizerFast.from_pretrained(config.base_model_name_or_path)

    qa_model = pipeline(task="question-answering", model=model, tokenizer=tokenizer)
    return(qa_model(question=question, context=context))

def vanilla_infer(question, context, MODEL_PATH):
    qa_model = pipeline(task="question-answering", model=MODEL_PATH, tokenizer=MODEL_PATH)
    return(qa_model(question=question, context=context))


if __name__ == "__main__":
    MODEL_PATH = "./models/qa-lora-bert-base"
    LORA = True

    question = "Where is the github link?"
    context = "You can find the github link for this video in the description."
    

    if LORA:
        print(lora_infer(question, context, MODEL_PATH))
    else:
        print(vanilla_infer(question, context, MODEL_PATH))
