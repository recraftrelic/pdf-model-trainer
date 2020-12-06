from pathlib import Path
from pdfminer.high_level import extract_text

import re
import json
import docx2txt

TRAIN_DATA = []
TRAIN_META_DATA = {
    "PERSON": {
        "REPLACE_FROM": "John Smith",
        "FILE": "./sentences/person.txt"
    },
    "FN": {
        "REPLACE_FROM": "James",
        "FILE": "./sentences/first_name.txt"
    },
    "LN": {
        "REPLACE_FROM": "Lee",
        "FILE": "./sentences/last_name.txt"
    },
}


def generate_sentences(file_name, replace_from, replace_with):
    with open(Path(file_name), "r") as sentences_file:
        sentences = sentences_file.read().split("\n")
        # filter empty lines
        sentences = [x for x in sentences if len(x) > 0]
        modified_sentences = []

        for sentence in sentences:
            modified_sentences.append(sentence.replace(replace_from, replace_with))

    return modified_sentences


def train():
    with open(Path("./train.json")) as train_file:
        data = json.load(train_file)

        for resume_record in data:
            if resume_record["resume_file"].find(".pdf") != -1:
                resume_text =  extract_text(Path(resume_record["resume_file"]))
            elif resume_record["resume_file"].find(".doc") != -1 or resume_record["resume_file"].find(".docx") != -1:
                resume_text = docx2txt.process(Path(resume_record["resume_file"]))
            entities = resume_record["entities"]

            for entity_key in entities:
                if entity_key in TRAIN_META_DATA:
                    sentences_file_path = TRAIN_META_DATA[entity_key].FILE
                    replace_from = TRAIN_META_DATA[entity_key].REPLACE_FROM

                sentences = generate_sentences(sentences_file_path, replace_from, entities[entity_key])
                for sentence in sentences:
                    indices = re.finditer(entities[entity_key], sentence)
                    indices_locations = []
                    for index in indices:
                        indices_locations.append([index.start(), index.end(), entity_key])

                    TRAIN_DATA.append(
                        {
                            "train_text": sentence,
                            "entities": indices_locations
                        }
                    )

                if entity_key == "PERSON":
                    indices = re.finditer(entities[entity_key], resume_text)

                    indices_locations = []

                    for index in indices:
                        indices_locations.append([index.start(), index.end(), entity_key])

                    TRAIN_DATA.append(
                        {
                            "train_text": resume_text,
                            "entities": indices_locations
                        }
                    )

    with open(Path("./train_data.json"), "w") as train_data_file:
        json.dump(TRAIN_DATA, train_data_file)


if __name__ == '__main__':
    train()
