from pathlib import Path
from spacy.util import minibatch, compounding

import plac
import random
import warnings
import spacy
import json

TRAIN_DATA = []


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    model_path=("Model path", "option", "mp", Path),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
    train_data_file_path=("Raw train data in json format", "option", "t", Path)
)
def main(model=None, model_path=None, output_dir=None, n_iter=100, train_data_file_path=None):
    if model or model_path is not None:
        if model is not None:
            nlp = spacy.load(model)
            print("Loaded model '%s'" % model)
        elif model_path is not None:
            nlp = spacy.load(Path(model_path))
            print("Loaded model '%s'" % model_path)
    else:
        nlp = spacy.load('en')
        print("Created blank 'en' model")

    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pip(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")

    if train_data_file_path is not None:
        train_data_file_path = Path(train_data_file_path)
        if not train_data_file_path.exists():
            train_data_file_path.touch()

        with open(train_data_file_path) as train_data_file:
            train_data_json = json.load(train_data_file)

            for training_record in train_data_json:
                train_text = training_record["train_text"]
                entities = []

                for entity in training_record["entities"]:
                    entities.append(
                        (entity[0], entity[1], entity[2])
                    )

                print(entities)

                TRAIN_DATA.append(
                    (train_text, {"entities": entities})
                )

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        print(model, model_path)

        if model and model_path is None:
            optimizer = nlp.begin_training()
        else:
            optimizer = nlp.entity.create_optimizer()

        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,
                    annotations,
                    sgd=optimizer,
                    drop=0.5,
                    losses=losses
                )

                print("Losses", losses)

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])


if __name__ == "__main__":
    plac.call(main)
