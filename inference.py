import timeit
from peft import PeftConfig, PeftModel
from transformers import BertForQuestionAnswering, BertTokenizerFast, pipeline


def lora_infer(question, context, MODEL_PATH):
    config = PeftConfig.from_pretrained(MODEL_PATH)
    model = PeftModel.from_pretrained(BertForQuestionAnswering.from_pretrained(config.base_model_name_or_path), MODEL_PATH)
    tokenizer = BertTokenizerFast.from_pretrained(config.base_model_name_or_path)

    start = timeit.default_timer()
    qa_model = pipeline(task="question-answering", model=model, tokenizer=tokenizer)
    stop = timeit.default_timer()
    print(f"Inference Time: {stop-start:.2f}s")
    return(qa_model(question=question, context=context))

def vanilla_infer(question, context, MODEL_PATH):
    model = BertForQuestionAnswering.from_pretrained(MODEL_PATH)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    start = timeit.default_timer()
    qa_model = pipeline(task="question-answering", model=model, tokenizer=tokenizer)
    stop = timeit.default_timer()
    print(f"Inference Time: {stop-start:.2f}s")
    return(qa_model(question=question, context=context))


if __name__ == "__main__":
    MODEL_PATH = "./models/qa-bert-base"
    LORA = False

    question = "Where is the github link?"
    context = "You can find the github link for this video in the description."
    

    if LORA:
        print(lora_infer(question, context, MODEL_PATH))
    else:
        print(vanilla_infer(question, context, MODEL_PATH))
