# This file contains commands which load datasets from kaggle to a local folder, and return it as a vector of tuples of (query, context, answer).

import os
import zipfile
import pandas
import random

import random_boolean_string

LOCAL_FOLDER = "./generated/datasets"

MATH_QA_DATASET_PATH = "thedevastator/dataset-for-solving-math-word-problems"
MATH_QA_DATASET_ZIP = "dataset-for-solving-math-word-problems.zip"
MATH_QA_LOCAL = "/mathqa"

SENTIMENTS_AND_EMOTIONS_DATASET_PATH = "ankitkumar2635/sentiment-and-emotions-of-tweets"
SENTIMENTS_AND_EMOTIONS_DATASET_ZIP = "sentiment-and-emotions-of-tweets.zip"
SENTIMENTS_AND_EMOTIONS_LOCAL = "/sentiment_emotions"

CHATGPT_REVIEWS_DATASET_PATH = "saloni1712/chatgpt-app-reviews"
CHATGPT_REVIEWS_DATASET_ZIP = "chatgpt-app-reviews.zip"
CHATGPT_REVIEWS_LOCAL = "/chatgpt_reviews"

MCDONALD_REVIEWS_DATASET_PATH = "nelgiriyewithana/mcdonalds-store-reviews"
MCDONALD_REVIEWS_DATASET_ZIP = "mcdonalds-store-reviews.zip"
MCDONALD_REVIEWS_LOCAL = "/mcdonald_reviews"

FAKE_V_REAL_NEWS_DATASET_PATH = "clmentbisaillon/fake-and-real-news-dataset"
FAKE_V_REAL_NEWS_DATASET_ZIP = "fake-and-real-news-dataset.zip"
FAKE_V_REAL_NEWS_LOCAL = "/fake_v_real_news"

HINDI_ENGLISH_TRANSLATION_DATASET_PATH = "vaibhavkumar11/hindi-english-parallel-corpus"
HINDI_ENGLISH_TRANSLATION_DATASET_ZIP = "hindi-english-parallel-corpus.zip"
HINDI_ENGLISH_TRANSLATION_LOCAL = "/hindi_english_translation"

# Cleab generated folder
def clear_All_Datasets():
    print("Removing all downloaded datasets")
    os.system("rm -rf " + LOCAL_FOLDER )

# Load a dataset to a specified folder, and unzip it.
def Load_Dataset(dataset_path, local_path, dataset_zip, do_unzip):
    os.system("kaggle datasets download " + dataset_path + " -p " + local_path)
    assert os.path.isfile(local_path + "/" + dataset_zip), "Load_Dataset failed to load"
    if do_unzip:
        with zipfile.ZipFile(local_path + "/" + dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(local_path)

# Load MathQA - received number between 1 and 7 (applies %8 on input - and asserts the result is non-zero integer)
# If set_num%2 = 1, uses the training dataset.
# If (set_num/2)%2 = 1, uses the validation dataset.
# If (set_num/4)%2 = 1, uses the test dataset.
# Datasets are appended together. set_num = 7 results in the combination of all datasets.
def Load_MathQA(set_num,do_load):
    print("Getting MathQA contents")
    assert (isinstance(set_num, int) and (set_num % 8 != 0)), "Load_MathQA received invalid set_num! {}".format(set_num)
    
    if do_load:
        print("Loading MathQA to local")
        Load_Dataset(MATH_QA_DATASET_PATH, LOCAL_FOLDER + MATH_QA_LOCAL, MATH_QA_DATASET_ZIP, True)
    
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
    print("Getting Sentiment & Emotions Labelled Tweets")
    
    if do_load:
        print("Loading Sentiment & Emotions Labelled Tweets to local")
        Load_Dataset(SENTIMENTS_AND_EMOTIONS_DATASET_PATH, LOCAL_FOLDER + SENTIMENTS_AND_EMOTIONS_LOCAL, SENTIMENTS_AND_EMOTIONS_DATASET_ZIP, True)
    
    # sentiments set
    usecols = ["Text","sentiment","sentiment_score"]
    if (to_emotion):
        # emotions set
        usecols = ["Text","emotion","emotion_score"]
        
    dataset = pandas.read_csv(LOCAL_FOLDER + SENTIMENTS_AND_EMOTIONS_LOCAL+"/sentiment-emotion-labelled_Dell_tweets.csv", usecols=usecols)

    dataset_vec = []

    for i in range(len(dataset[usecols[0]])):
        if (variant_score == 0): # classification given text
            item = {"query":dataset[usecols[0]][i],"context":"","answer":dataset[usecols[1]][i]}
        else: # score given text and classification
            item = {"query":dataset[usecols[0]][i],"context":dataset[usecols[1]][i],"answer":dataset[usecols[2]][i]}
        dataset_vec.append(item)

    print("Sentiment & Emotions Labelled Tweets lines fetched: {}".format(len(dataset_vec)))
    return dataset_vec

# Load ChatGPT App Reviews - title and text to rating
def Load_ChatGPT_Reviews(do_load):
    print("Getting ChatGPT App Reviews")
    
    if do_load:
        print("Loading ChatGPT App Reviews to local")
        Load_Dataset(CHATGPT_REVIEWS_DATASET_PATH, LOCAL_FOLDER + CHATGPT_REVIEWS_LOCAL, CHATGPT_REVIEWS_DATASET_ZIP, True)
    
    usecols = ["title","review","rating"]
        
    dataset = pandas.read_csv(LOCAL_FOLDER + CHATGPT_REVIEWS_LOCAL+"/chatgpt_reviews.csv", usecols=usecols)

    dataset_vec = []

    for i in range(len(dataset[usecols[0]])):
        item = {"query":dataset[usecols[0]][i],"context":dataset[usecols[1]][i],"answer":dataset[usecols[2]][i]}
        dataset_vec.append(item)

    print("ChatGPT App Reviews lines fetched: {}".format(len(dataset_vec)))
    return dataset_vec

# Load McDonald's Store Reviews - text to rating
def Load_McDonald_Reviews(do_load):
    print("Getting McDonald's Store Reviews")
    
    if do_load:
        print("Loading McDonald's Store Reviews to local")
        Load_Dataset(MCDONALD_REVIEWS_DATASET_PATH, LOCAL_FOLDER + MCDONALD_REVIEWS_LOCAL, MCDONALD_REVIEWS_DATASET_ZIP, True)
    
    usecols = ["review","rating"]

    dataset = pandas.read_csv(LOCAL_FOLDER + MCDONALD_REVIEWS_LOCAL+"/McDonald_s_Reviews.csv", usecols=usecols, encoding='latin-1')

    dataset_vec = []

    for i in range(len(dataset[usecols[0]])):
        item = {"query":dataset[usecols[0]][i],"context":"","answer":dataset[usecols[1]][i][0]} # in dataset the rating is "n star", take only the "n".
        dataset_vec.append(item)

    print("McDonald's Store Reviews fetched: {}".format(len(dataset_vec)))
    return dataset_vec

# Load Fake and Real News - title and text to real or fake
def Load_Fake_and_Real_News(do_load):
    print("Getting Fake and Real News")

    if do_load:
        print("Loading Fake and Real News to local")
        Load_Dataset(FAKE_V_REAL_NEWS_DATASET_PATH, LOCAL_FOLDER + FAKE_V_REAL_NEWS_LOCAL, FAKE_V_REAL_NEWS_DATASET_ZIP, True)
    
    usecols = ["title","text"]
        
    fake_dataset = pandas.read_csv(LOCAL_FOLDER + FAKE_V_REAL_NEWS_LOCAL+"/Fake.csv", usecols=usecols)
    real_dataset = pandas.read_csv(LOCAL_FOLDER + FAKE_V_REAL_NEWS_LOCAL+"/True.csv", usecols=usecols)

    dataset_vec = []

    for i in range(len(fake_dataset[usecols[0]])):
        item = {"query":fake_dataset[usecols[0]][i],"context":fake_dataset[usecols[1]][i],"answer":"fake"}
        dataset_vec.append(item)
        
    for i in range(len(real_dataset[usecols[0]])):
        item = {"query":real_dataset[usecols[0]][i],"context":real_dataset[usecols[1]][i],"answer":"real"}
        dataset_vec.append(item)

    print("Fake and Real News fetched: {}".format(len(dataset_vec)))
    return dataset_vec

# Use the random string generation code to create a dataset and use it
def Load_Boolean_String(min_string,max_string,probability_true,num_of_strings):
    print("Generating random boolean strings")
    
    dataset_vec = []

    for i in range(num_of_strings):
        complexity = random.randint(min_string,max_string)
        boolean_string, boolean_result = random_boolean_string.get_random_boolean_with_probability(complexity,probability_true)
        answer_string = "TRUE" if boolean_result else "FALSE"
        item = {"query":boolean_string,"context":"","answer":answer_string}
        dataset_vec.append(item)

    print("Boolean strings generated: {}".format(len(dataset_vec)))
    return dataset_vec

# Hindi - English Translation dataset from IIT Bombay
def Load_Hindi_English_Translation(do_load,to_english):
    print("Getting Hindi - English Translation")
    
    if do_load:
        print("Loading Hindi - English Translation to local")
        Load_Dataset(HINDI_ENGLISH_TRANSLATION_DATASET_PATH, LOCAL_FOLDER + HINDI_ENGLISH_TRANSLATION_LOCAL, HINDI_ENGLISH_TRANSLATION_DATASET_ZIP, True)
    
    usecols = ["hindi","english"]

    dataset = pandas.read_csv(LOCAL_FOLDER + HINDI_ENGLISH_TRANSLATION_LOCAL+"/hindi_english_parallel.csv", usecols=usecols)

    dataset_vec = []

    for i in range(len(dataset[usecols[0]])):
        if to_english:
            item = {"query":dataset[usecols[0]][i],"context":"","answer":dataset[usecols[1]][i]}
        else:
            item = {"query":dataset[usecols[1]][i],"context":"","answer":dataset[usecols[0]][i]}
        dataset_vec.append(item)

    print("Hindi - English Translation fetched: {}".format(len(dataset_vec)))
    return dataset_vec

if __name__ == "__main__":
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
    datasetGPT = Load_ChatGPT_Reviews(True)
    for row in datasetGPT[0:min(len(datasetGPT),5)]:
        print(row)
   
    print("McDonald's Store Reviews examples:")
    datasetMCD = Load_McDonald_Reviews(True)
    for row in datasetMCD[0:min(len(datasetMCD),5)]:
        print(row)
   
    print("Fake and Real News examples:")
    datasetFVR = Load_Fake_and_Real_News(True)
    for row in datasetFVR[0:min(len(datasetFVR),5)]:
        print(row)
    
    print("Boolean strings examples:")
    datasetBOOL = Load_Boolean_String(7,15,0.5,5)
    for row in datasetBOOL:
        print(row)
    
    print("Hindi - English Translation examples:")
    datasetHIN_ENG = Load_Hindi_English_Translation(True,False)
    for row in datasetHIN_ENG[0:min(len(datasetHIN_ENG),5)]:
        print(row)
    
    os.system("echo clearing loaded datasets")
    clear_All_Datasets()
    os.system("echo done clearing loaded datasets")