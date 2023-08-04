## Fine-Tune BERT For Question Answering

![Sample Result](./assets/qa-sample.png)

This repository contains an easy-to-use and understand code to fine-tune BERT for Question-Answering(Q&A). Sample training was made by using [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/). Dataset preparation part was inspired from [HuggingFace Tutorial](https://huggingface.co/transformers/v3.2.0/custom_datasets.html#question-answering-with-squad-2-0).

Above example demonstrates a sample context, question and answer. 

### YouTube Tutorial
This repository also contains a corresponding YouTube tutorial with the title **Fine-Tune BERT For Question-Answering(Q&A) - PyTorch & HuggingFace**

[![Thumbnail](./assets/youtube-thumbnail.png)](https://www.youtube.com/watch?v=PikqVppe408&t=9s)

### Project Structure
Project structured as follows:
```
.
└── src/
    ├── squad_dataset.py
    ├── main.py
    ├── inference.py
    ├── data/
    │   └── SQuAD.json
    └── models/
```

`squad_dataset.py` creates the PyTorch dataset. `main.py` file contains the training loop. `inference.py` contains necessary functions to easly run an inference.

`models/` directory is to save and store the trained models.

`data/` directory contains the data you're going to train on in `.json` format.

### Pre-Trained Model
You can download a sample pre-trained model from [here](https://drive.google.com/file/d/1aIcI_9RRWVUJHts5ZgsKDuH4HjVFe467/view?usp=sharing). Put the model into the `models/` directory. Note that this sample model was trained on 20000 samples of the whole dataset.

### Inference
`inference.py` file provides easy pipeline to use. Change the `context` and `question` variables based on your need. Change the `model` and `tokenizer` arguments inside the pipeline to point to your trained model.

### Training
In order to train the model you must run the command `python main.py`. File has hyperparameters of `LEARNING_RATE`, `BATCH_SIZE` and `EPOCHS`. You can change them as you like.

You must give your data directory, the directory you want to save your model to and your base model to `DATA_PATH` and `MODEL_SAVE_PATH` and `MODEL_PATH`variables in the `main.py` file. By default `bert-base-uncased` selected to serve as the base model.

By the end of the training your model will be saved into the `MODEL_SAVE_PATH`.
