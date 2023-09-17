# This file contains commands which take database-specialized vectors of tuples of (query, context, answer), and add a (Question, ...) prefix matching the database.

def parse_MathQA(in_vec):
    for item in in_vec:
        item["question"] = "pick the best answer to the math question from the five possible answers provided, from a to e"
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