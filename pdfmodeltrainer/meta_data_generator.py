from os import listdir
from os.path import isfile, join, splitext
from train_data_generator import generate_train_data
from pathlib import Path

import json
import shutil
import plac

CURRENT_PATH = Path(__file__).parent.absolute()
train_json = []


@plac.annotations(
    resumes_folder=("Path to resumes folder", "option", "f", Path),
    resumes_done_folder=("Path to output done resumes", "option", "o", Path)
)
def main(resumes_folder=Path("./resumes"), resumes_done_folder=Path("./resumes_done")):
    files = [f for f in listdir(resumes_folder) if isfile(join(resumes_folder, f))]

    print(files)

    for file in files:
        if file.find(".DS_Store") == -1:
            filename = splitext(file)[0]

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
                "resume_file": join(resumes_folder, file)
            }

            if len(filename.split("-")) > 1:
                LN = filename.split("-")[1]
                train_record["entities"]["LN"] = LN

            train_json.append(train_record)

    with open(Path(join(CURRENT_PATH, "./train_meta.json")), "w") as train_json_file:
        json.dump(train_json, train_json_file)

    # generate train json
    generate_train_data()

    for file in files:
        shutil.move(join(resumes_folder, file), join(resumes_done_folder, file))
        print("Moved", file)

    # main


if __name__ == '__main__':
    plac.call(main)
