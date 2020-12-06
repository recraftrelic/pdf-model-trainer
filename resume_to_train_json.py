from os import listdir
from os.path import isfile, join, splitext
from create_train_json import train

import json
import shutil

provided_path = "./resumes"
train_json = []

files = [f for f in listdir(provided_path) if isfile(join(provided_path, f))]

for file in files:
    if file.find(".DS_Store") == -1:
        filename = splitext(file)[0]

        file_parts = []

        if filename.find("--") > -1:
            file_parts = filename.split("--")
            file_parts = file_parts[0:len(file_parts) - 1]
        else:
            file_parts = [filename]

        if len(file_parts) > 0:
            filename = file_parts[0]

        PERSON = filename.replace("-", " ")
        FN = filename.split("-")[0]

        train_record = {
            "entities": {
                "FN": FN,
                "PERSON": PERSON,
            },
            "resume_file": join(provided_path, file)
        }

        if len(filename.split("-")) > 1:
            LN = filename.split("-")[1]
            train_record["entities"]["LN"] = LN

        train_json.append(train_record)

with open("./train.json", "w") as train_json_file:
    json.dump(train_json, train_json_file)

train()

for file in files:
    shutil.move(join(provided_path, file), join("./resumes_done", file))
    print("Moved", file)
