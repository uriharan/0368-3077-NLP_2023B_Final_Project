# This file contains commands which load datasets from kaggle to a local folder, and return it as a vector of tuples of (query, context, answer).

import os
import zipfile
import pandas

LOCAL_FOLDER = "./generated/datasets"

MATH_QA_DATASET_PATH = "thedevastator/dataset-for-solving-math-word-problems"
MATH_QA_DATASET_NAME = "dataset-for-solving-math-word-problems.zip"
MATH_QA_LOCAL = "/mathqa"

# Load a dataset to a specified folder, and unzip it.
def Load_Dataset(dataset_path, local_path, dataset_name, do_unzip):
    os.system("kaggle datasets download " + dataset_path + " -p " + local_path)
    if do_unzip:
        with zipfile.ZipFile(local_path + dataset_name, 'r') as zip_ref:
            zip_ref.extractall(local_path)

# Load MathQA - received number between 1 and 7 (applies %8 on input - and asserts the result is non-zero integer)
# If set_num%2 = 1, uses the training dataset.
# If (set_num/2)%2 = 1, uses the validation dataset.
# If (set_num/4)%2 = 1, uses the test dataset.
# Datasets are appended together. set_num = 7 results in the combination of all datasets.
def Load_MathQA(set_num,do_load):
    print("Fetching MathQA contents")
    assert (isinstance(set_num, int) and (set_num % 8 != 0)), "Load_MathQA received invalid set_num! {}".format(set_num)
    
    if do_load:
        print("Loading MathQA to local")
        Load_Dataset(MATH_QA_DATASET_PATH, LOCAL_FOLDER + MATH_QA_LOCAL, MATH_QA_DATASET_NAME, True)
    
    usecols = ["Problem","options","correct"]
    math_qa_vec = []

    # Train set
    if (set_num%2 == 1):
        train_qa = pandas.read_csv(LOCAL_FOLDER + MATH_QA_LOCAL+"/train.csv", usecols=usecols)
        for i in range(len(train_qa['Problem'])):
            math_qa_vec.append((train_qa['Problem'][i],train_qa['options'][i],train_qa['correct'][i]))

    # Validation set
    if ((set_num/2)%2 == 1):
        validation_qa = pandas.read_csv(LOCAL_FOLDER + MATH_QA_LOCAL+"/validation.csv", usecols=usecols)
        for i in range(len(validation_qa['Problem'])):
            math_qa_vec.append((validation_qa['Problem'][i],validation_qa['options'][i],validation_qa['correct'][i]))

    # Test set
    if ((set_num/4)%2 == 1):
        test_qa = pandas.read_csv(LOCAL_FOLDER + MATH_QA_LOCAL+"/test.csv", usecols=usecols)
        for i in range(len(test_qa['Problem'])):
            math_qa_vec.append((test_qa['Problem'][i],test_qa['options'][i],test_qa['correct'][i]))

    print("MathQA lines fetched: {}".format(len(math_qa_vec)))
    return math_qa_vec

if __name__ == "main":
    dataset = Load_MathQA(4,True)
    for row in dataset[0:min(len(dataset),10)]:
        print(row)