# This file includes code designed to combine the dataset_loaders.py, dataset_parsers.py, and politeness_prefix.py into huggingface-compatible examples and answers.
# First - load the query-context-answer vectors of dictionaries, and truncate it to the wanted length.
# Second - add the prefix and dataset-based question to get a prefix-question-query-context-answer vector tuple.
# Third - combine the first several vectors, get text-answer

import os

import dataset_loaders
import dataset_parsers
import politeness_prefix

def assemble_MathQA(num_of_results,set_num,do_load,politeness,constant_variant,word_variant,min_index):
    # Get vector
    results_init = dataset_loaders.Load_MathQA(set_num,do_load)
    results_truncated = results_init[min_index:min(num_of_results+min_index,len(results_init)-1)]
    results_parsed = dataset_parsers.parse_MathQA(results_truncated)
    results_with_prefix = politeness_prefix.add_prefix(results_parsed,politeness,constant_variant,word_variant)
    
    data_arranged = {"text":[],"answer":[]}

    # Concatenate prefix-question-query-context
    for i in range(len(results_with_prefix)):
        item = results_with_prefix[i]['prefix'] + " " + results_with_prefix[i]['question'] + ". Question: \"" + results_with_prefix[i]['query'] + "\", Options: \"" + results_with_prefix[i]['context'] + "\"."
        data_arranged["text"].append(item)
        data_arranged["answer"].append(results_with_prefix[i]['answer'])
    
    return data_arranged

def assemble_SentimentsAndEmotions(num_of_results,to_emotion,do_load,variant_score,politeness,constant_variant,word_variant,min_index):
    # Get vector
    results_init = dataset_loaders.Load_SentimentsAndEmotions(to_emotion,do_load,variant_score)
    results_truncated = results_init[min_index:min(num_of_results+min_index,len(results_init)-1)]
    results_parsed = dataset_parsers.parse_SentimentsAndEmotions(results_truncated,to_emotion,variant_score)
    results_with_prefix = politeness_prefix.add_prefix(results_parsed,politeness,constant_variant,word_variant)

    data_arranged = {"text":[],"answer":[]}

    # Concatenate prefix-question-query-context
    for i in range(len(results_with_prefix)):
        if not variant_score:
            item = results_with_prefix[i]['prefix'] + " " + results_with_prefix[i]['question'] + "? \"" + results_with_prefix[i]['query'] + "\""
        else:
            item = results_with_prefix[i]['prefix'] + " " + results_with_prefix[i]['question'] + "? Review: \"" + results_with_prefix[i]['query'] + "\", Estimation: \"" + results_with_prefix[i]['context'] + "\""
        data_arranged["text"].append(item)
        data_arranged["answer"].append(results_with_prefix[i]['answer'])
    
    return data_arranged

def assemble_ChatGPT_Reviews(num_of_results,do_load,politeness,constant_variant,word_variant,min_index):
    results_init = dataset_loaders.Load_ChatGPT_Reviews(do_load)
    results_truncated = results_init[min_index:min(num_of_results+min_index,len(results_init)-1)]
    results_parsed = dataset_parsers.parse_ChatGPT_Reviews(results_truncated)
    results_with_prefix = politeness_prefix.add_prefix(results_parsed,politeness,constant_variant,word_variant)

    data_arranged = {"text":[],"answer":[]}

    # Concatenate prefix-question-query-context
    for i in range(len(results_with_prefix)):
        item  =  results_with_prefix[i]['prefix'] + " " + results_with_prefix[i]['question'] + "? Title: \"" + results_with_prefix[i]['query'] + "\", Content: \"" + results_with_prefix[i]['context'] + "\""
        data_arranged["text"].append(item)
        data_arranged["answer"].append(results_with_prefix[i]['answer'])
    return data_arranged

def assemble_McDonald_Reviews(num_of_results,do_load,politeness,constant_variant,word_variant,min_index):
    results_init = dataset_loaders.Load_McDonald_Reviews(do_load)
    results_truncated = results_init[min_index:min(num_of_results+min_index,len(results_init)-1)]
    results_parsed = dataset_parsers.parse_McDonald_Reviews(results_truncated)
    results_with_prefix = politeness_prefix.add_prefix(results_parsed,politeness,constant_variant,word_variant)
    
    data_arranged = {"text":[],"answer":[]}

    # Concatenate prefix-question-query
    for i in range(len(results_with_prefix)):
        item  =  results_with_prefix[i]['prefix'] + " " + results_with_prefix[i]['question'] + "? \"" + results_with_prefix[i]['query'] + "\""
        data_arranged["text"].append(item)
        data_arranged["answer"].append(results_with_prefix[i]['answer'])
    return data_arranged

def assemble_Fake_and_Real_News(num_of_results,do_load,politeness,constant_variant,word_variant,min_index):
    results_init = dataset_loaders.Load_Fake_and_Real_News(do_load)
    results_truncated = results_init[min_index:min(num_of_results+min_index,len(results_init)-1)]
    results_parsed = dataset_parsers.parse_Fake_and_Real_News(results_truncated)
    results_with_prefix = politeness_prefix.add_prefix(results_parsed,politeness,constant_variant,word_variant)

    data_arranged = {"text":[],"answer":[]}

    # Concatenate prefix-question-query-context
    for i in range(len(results_with_prefix)):
        item  =  results_with_prefix[i]['prefix'] + " " + results_with_prefix[i]['question'] + ": Title: \"" + results_with_prefix[i]['query'] + "\", Text: \"" + results_with_prefix[i]['context'] + "\""
        data_arranged["text"].append(item)
        data_arranged["answer"].append(results_with_prefix[i]['answer'])
    return data_arranged

def assemble_Boolean_String(num_of_results,min_string,max_string,probability_true,politeness,constant_variant,word_variant):
    results_init = dataset_loaders.Load_Boolean_String(min_string,max_string,probability_true,num_of_results)
    # Already set to correct length - no truncation
    results_parsed = dataset_parsers.parse_Boolean_String(results_init)
    results_with_prefix = politeness_prefix.add_prefix(results_parsed,politeness,constant_variant,word_variant)

    data_arranged = {"text":[],"answer":[]}

    # Concatenate prefix-question-query
    for i in range(len(results_with_prefix)):
        item  =  results_with_prefix[i]['prefix'] + " " + results_with_prefix[i]['question'] + ": \"" + results_with_prefix[i]['query'] + "\""
        data_arranged["text"].append(item)
        data_arranged["answer"].append(results_with_prefix[i]['answer'])
    return data_arranged

def assemble_English_Translation(num_of_results,do_load,to_english,other_lang,politeness,constant_variant,word_variant,min_index):
    assert other_lang == "hindi" or other_lang == "french", "Unsupported language {other_lang}"
    if other_lang == "hindi":
        results_init = dataset_loaders.Load_Hindi_English_Translation(do_load,to_english)
    else:
        results_init = dataset_loaders.Load_English_French_Translation(do_load,to_english)
    results_truncated = results_init[min_index:min(num_of_results+min_index,len(results_init)-1)]
    results_parsed = dataset_parsers.parse_English_Translation(results_truncated,to_english,other_lang)
    results_with_prefix = politeness_prefix.add_prefix(results_parsed,politeness,constant_variant,word_variant)

    data_arranged = {"text":[],"answer":[]}

    # Concatenate prefix-question-query
    for i in range(len(results_with_prefix)):
        item  =  results_with_prefix[i]['prefix'] + " " + results_with_prefix[i]['question'] + ": \"" + results_with_prefix[i]['query'] + "\""
        data_arranged["text"].append(item)
        data_arranged["answer"].append(results_with_prefix[i]['answer'])
    return data_arranged

if __name__ == "__main__":
    os.system("echo entered data_assembler as main")

    print("MathQA examples:")
    datasetQA = assemble_MathQA(5,4,True,-3,False,0)
    for row in range(len(datasetQA["text"])):
        print(datasetQA["text"][row] + " ||| " + datasetQA["answer"][row])

    print("S&E examples:")
    datasetSE = assemble_SentimentsAndEmotions(5,False,True,True,-2,False,0)
    for row in range(len(datasetSE["text"])):
        print(datasetSE["text"][row] + " ||| " + str(datasetSE["answer"][row]))
    
    print("GPT Reviews examples:")
    datasetGPT = assemble_ChatGPT_Reviews(5,True,-1,False,0)
    for row in range(len(datasetGPT["text"])):
        print(datasetGPT["text"][row] + " ||| " + str(datasetGPT["answer"][row]))
   
    print("McDonald's Store Reviews examples:")
    datasetMCD = assemble_McDonald_Reviews(5,True,1,False,0)
    for row in range(len(datasetMCD["text"])):
        print(datasetMCD["text"][row] + " ||| " + datasetMCD["answer"][row])
   
    print("Fake and Real News examples:")
    datasetFVR = assemble_Fake_and_Real_News(5,True,2,False,0)
    for row in range(len(datasetFVR["text"])):
        print(datasetFVR["text"][row] + " ||| " + datasetFVR["answer"][row])

    print("Boolean strings examples:")
    datasetBOOL = assemble_Boolean_String(5,7,15,0.5,3,False,0)
    for row in range(len(datasetBOOL["text"])):
        print(datasetBOOL["text"][row] + " ||| " + datasetBOOL["answer"][row])
    
    print("Hindi - English Translation examples:")
    datasetHIN_ENG = assemble_English_Translation(5,True,False,"hindi",3,True,2)
    for row in range(len(datasetHIN_ENG["text"])):
        print(datasetHIN_ENG["text"][row] + " ||| " + datasetHIN_ENG["answer"][row])
    
    print("English - French Translation examples:")
    datasetENG_FRE = assemble_English_Translation(5,True,True,"french",-3,True,2)
    for row in range(len(datasetENG_FRE["text"])):
        print(datasetENG_FRE["text"][row] + " ||| " + datasetENG_FRE["answer"][row])

    os.system("echo clearing loaded datasets")
    dataset_loaders.clear_All_Datasets()
    os.system("echo done clearing loaded datasets")