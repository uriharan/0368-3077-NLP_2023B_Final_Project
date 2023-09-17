# This file contains commands which load datasets from kaggle to a local folder, and return it as a vector of tuples of (query, context, answer).

import os
import zipfile
import pandas

LOCAL_FOLDER = "./generated/datasets"

MATH_QA_DATASET_PATH = "thedevastator/dataset-for-solving-math-word-problems"
MATH_QA_DATASET_NAME = "dataset-for-solving-math-word-problems.zip"
MATH_QA_LOCAL = "/mathqa"

SENTIMENTS_AND_EMOTIONS_DATASET_PATH = "ankitkumar2635/sentiment-and-emotions-of-tweets"
SENTIMENTS_AND_EMOTIONS_DATASET_NAME = "sentiment-and-emotions-of-tweets.zip"
SENTIMENTS_AND_EMOTIONS_LOCAL = "sentiment_emotions/"

CHATGPT_REVIEWS_DATASET_PATH = "saloni1712/chatgpt-app-reviews"
CHATGPT_REVIEWS_DATASET_NAME = "chatgpt-app-reviews.zip"
CHATGPT_REVIEWS_LOCAL = "chatgpt_reviews/"

# Cleab generated folder
def clear_All_Datasets():
    os.system("rm -rf " + LOCAL_FOLDER )

# Load a dataset to a specified folder, and unzip it.
def Load_Dataset(dataset_path, local_path, dataset_name, do_unzip):
    os.system("kaggle datasets download " + dataset_path + " -p " + local_path)
    assert os.path.isfile(local_path + "/" + dataset_name), "Load_Dataset failed to load"
    if do_unzip:
        with zipfile.ZipFile(local_path + "/" + dataset_name, 'r') as zip_ref:
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

    dataset = []

    # Train set
    if (set_num%2 == 1):
        dataset = pandas.read_csv(LOCAL_FOLDER + MATH_QA_LOCAL+"/train.csv", usecols=usecols)

    # Validation set
    if ((set_num/2)%2 == 1):
        dataset = pandas.read_csv(LOCAL_FOLDER + MATH_QA_LOCAL+"/validation.csv", usecols=usecols)

    # Test set
    if ((set_num/4)%2 == 1):
        dataset = pandas.read_csv(LOCAL_FOLDER + MATH_QA_LOCAL+"/test.csv", usecols=usecols)


    for i in range(len(dataset[usecols[0]])):
        item = {"query":dataset[usecols[0]][i],"context":dataset[usecols[1]][i],"answer":dataset[usecols[2]][i]}
        math_qa_vec.append(item)

    print("MathQA lines fetched: {}".format(len(math_qa_vec)))
    return math_qa_vec

# Load Sentiment & Emotions Labelled Tweets - text to sentiment or to emotion
# variant_score - 0 for base classification, else for asking score given classification
def Load_SentimentsAndEmotions(to_emotion,do_load,variant_score):
    print("Sentiment & Emotions Labelled Tweets")
    
    if do_load:
        print("Loading Sentiment & Emotions Labelled Tweets to local")
        Load_Dataset(SENTIMENTS_AND_EMOTIONS_DATASET_PATH, LOCAL_FOLDER + SENTIMENTS_AND_EMOTIONS_LOCAL, SENTIMENTS_AND_EMOTIONS_DATASET_NAME, True)
    
    # sentiments set
    usecols = ["Text","sentiment","sentiment_score"]
    if (to_emotion):
        # emotions set
        usecols = ["Text","emotion","emotion_score"]
        
    dataset = pandas.read_csv(LOCAL_FOLDER + SENTIMENTS_AND_EMOTIONS_LOCAL+"/sentiment-emotion-labelled_Dell_tweets.csv", usecols=usecols)

    dataset_vec = []

    if (variant_score == 0): # classification given text
        for i in range(len(dataset[usecols[0]])):
            dataset_vec.append((dataset[usecols[0]][i],"",dataset[usecols[1]][i]))
    else: # score given text and classification
        for i in range(len(dataset[usecols[0]])):
            dataset_vec.append((dataset[usecols[0]][i],dataset[usecols[1]][i],dataset[usecols[2]][i]))

    for i in range(len(dataset[usecols[0]])):
        if (variant_score == 0):
            item = {"query":dataset[usecols[0]][i],"context":"","answer":dataset[usecols[1]][i]}
        else:
            item = {"query":dataset[usecols[0]][i],"context":dataset[usecols[1]][i],"answer":dataset[usecols[2]][i]}

    print("Sentiment & Emotions Labelled Tweets lines fetched: {}".format(len(dataset_vec)))
    return dataset_vec

# Load ChatGPT App Reviews - title and text to rating
def Load_ChatGPT_Reviews(do_load):
    print("ChatGPT App Reviews")
    
    if do_load:
        print("Loading ChatGPT App Reviews to local")
        Load_Dataset(CHATGPT_REVIEWS_DATASET_PATH, LOCAL_FOLDER + CHATGPT_REVIEWS_LOCAL, CHATGPT_REVIEWS_DATASET_NAME, True)
    
    # sentiments set
    usecols = ["title","review","rating"]
        
    dataset = pandas.read_csv(LOCAL_FOLDER + CHATGPT_REVIEWS_LOCAL+"/chatgpt_reviews.csv", usecols=usecols)

    dataset_vec = []

    for i in range(len(dataset[usecols[0]])):
        dataset_vec.append((dataset[usecols[0]][i],"",dataset[usecols[1]][i]))

    for i in range(len(dataset[usecols[0]])):
        item = {"query":dataset[usecols[0]][i],"context":dataset[usecols[1]][i],"answer":dataset[usecols[2]][i]}

    print("ChatGPT App Reviews lines fetched: {}".format(len(dataset_vec)))
    return dataset_vec

if __name__ == "main":
    os.system("echo entered dataset_loaders as main")
    print("MathQA examples:")
    datasetQA = Load_MathQA(4,True)
    for row in datasetQA[0:min(len(datasetQA),5)]:
        print(row)
    print("S&E examples:")
    datasetSE = Load_SentimentsAndEmotions(False,True,True)
    for row in datasetSE[0:min(len(datasetSE),5)]:
        print(row)
    print("GPT Reviews examples:")
    datasetGPT = Load_ChatGPT_Reviews()
    for row in datasetGPT[0:min(len(datasetGPT),5)]:
        print(row)