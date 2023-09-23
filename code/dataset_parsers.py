# This file contains commands which take database-specialized vectors of tuples of (query, context, answer), and add a (Question, ...) prefix matching the database.

def parse_MathQA(in_vec):
    for item in in_vec:
        item["question"] = "given the five possible answers provided, from a to e, pick the one which answers the math question correctly"
    return in_vec
    
def parse_SentimentsAndEmotions(in_vec,to_emotion,variant_score):
    for item in in_vec:
        if (not to_emotion and not variant_score):
            item["question"] = "given the following review, is it a positive, neutral, or negative review"
        elif (to_emotion and not variant_score):
            item["question"] = "given the following review, what is the best-fitting emotion to it from the list: joy, love, optimism, pessimism, trust, surprise, anticipation, sadness, anger, disgust, or fear"
        elif (not to_emotion and variant_score):
            item["question"] = "given the following review and the sentiment estimation, on a scale from 0 to 1, how fitting is the sentiment to the review"
        else:
            item["question"] = "given the following review and the emotion estimation, on a scale from 0 to 1, how fitting is the emotion to the review"
    return in_vec

def parse_ChatGPT_Reviews(in_vec):
    for item in in_vec:
        item["question"] = "given the following review, seperated into title and content, what is the review's given score, as an integer from 1 to 5"
    return in_vec

def parse_McDonald_Reviews(in_vec):
    for item in in_vec:
        item["question"] = "given the following review, what is the review's given score, as an integer from 1 to 5"
    return in_vec

def parse_Fake_and_Real_News(in_vec):
    for item in in_vec:
        item["question"] = "given the following news article, seperated into title and text, determine if the article is real or fake"
    return in_vec

def parse_Boolean_String(in_vec):
    for item in in_vec:
        item["question"] = "given the following boolean expression, evaulate whether its truth value is TRUE or FALSE"
    return in_vec


def parse_Hindi_English_Translation(in_vec,to_english):
    var_from = "hindi" if to_english else "english"
    var_to = "english" if to_english else "hindi"
    for item in in_vec:
        item["question"] = "given the following string in " + var_from + ", translate it to the " + var_to + " language"
    return in_vec