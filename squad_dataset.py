import torch
from torch.utils.data import Dataset
import json

class SquadDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        contexts, questions, answers = self.read_data(data_path)
        answers = self.add_end_idx(contexts, answers)

        encodings = tokenizer(contexts, questions, padding=True, truncation=True)
        self.encodings = self.update_start_end_positions(encodings, answers, tokenizer)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

    def read_data(self, path):
        with open(path, 'rb') as f:
            squad = json.load(f)

        contexts = []
        questions = []
        answers = []
        for group in squad['data']:
            for parag in group['paragraphs']:
                context = parag['context']
                for qa in parag['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)

        return contexts, questions, answers

    def add_end_idx(self, contexts, answers):
        for answer, context in zip(answers, contexts):
            gold_text = answer["text"]
            start_idx = answer["answer_start"]

            end_idx = start_idx + len(gold_text)

            if context[start_idx:end_idx] == gold_text:
                answer["answer_end"] = end_idx
            elif context[start_idx-1:end_idx-1] == gold_text:
                answer["answer_start"] = start_idx - 1
                answer["answer_end"] = end_idx - 1
            elif context[start_idx-2:end_idx-2] == gold_text:
                answer["answer_start"] = start_idx - 2
                answer["answer_end"] = end_idx - 2
        return answers

    def update_start_end_positions(self, encodings, answers, tokenizer):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i, answers[i]["answer_start"]))
            end_positions.append(encodings.char_to_token(i, answers[i]["answer_end"]-1))
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length
        encodings["start_positions"] = start_positions
        encodings["end_positions"] = end_positions

        return encodings
