# This file includes code designed to combine the dataset_loaders.py, dataset_parsers.py, and politeness_prefix.py into huggingface-compatible examples and answers.
# First - load the query-context-answer tuple of vectors, and truncate it to the wanted length.
# Second - add the prefix and dataset-based question to get a prefix-question-query-context-answer vector tuple.
# Third - combine the first several vectors, get text-answer

import os

import dataset_loaders
import dataset_parsers
import politeness_prefix

def assemble_MathQA(num_of_results,set_num,do_load,politeness,constant_variant,word_variant):
    # Get vector
    results_init = dataset_loaders.Load_MathQA(set_num,do_load)
    results_truncated = results_init[0:min(num_of_results,len(results_init)-1)]
    results_parsed = dataset_parsers.parse_MathQA(results_truncated)
    results_with_prefix = politeness_prefix.add_prefix(results_parsed,politeness,constant_variant,word_variant)
    
    data_arranged = ()

    # Concatenate prefix-question-query-context
    for i in range(len(results_with_prefix['query'])):
        item = results_with_prefix['prefix'][i] + " " + results_with_prefix['question'][i] + ". Question: \"" + results_with_prefix['query'] + "\", Options: \"" + results_with_prefix['context'] + "\"."
        data_arranged["text"].append(item)
        data_arranged["answer"].append(results_with_prefix['answer'][i])
    
    return data_arranged

def assemble_SentimentsAndEmotions(num_of_results,to_emotion,do_load,variant_score,politeness,constant_variant,word_variant):
    # Get vector
    results_init = dataset_loaders.Load_SentimentsAndEmotions(to_emotion,do_load,variant_score)
    results_truncated = results_init[0:min(num_of_results,len(results_init)-1)]
    results_parsed = dataset_parsers.parse_SentimentsAndEmotions(results_truncated,to_emotion,variant_score)
    results_with_prefix = politeness_prefix.add_prefix(results_parsed,politeness,constant_variant,word_variant)
    
    data_arranged = ()

    # Concatenate prefix-question-query-context
    for i in range(len(results_with_prefix['query'])):
        if not variant_score:
            item = results_with_prefix['prefix'][i] + " " + results_with_prefix['question'][i] + "? \"" + results_with_prefix['query'] + "\""
        else:
            item = results_with_prefix['prefix'][i] + " " + results_with_prefix['question'][i] + "? Review: \"" + results_with_prefix['query'] + "\", Estimation: \"" + results_with_prefix['context'] + "\""
        data_arranged["text"].append(item)
        data_arranged["answer"].append(results_with_prefix['answer'][i])
    
    return data_arranged

def assemble_ChatGPT_Reviews(num_of_results,do_load,politeness,constant_variant,word_variant):
    results_init = dataset_loaders.Load_ChatGPT_Reviews(do_load)
    results_truncated = results_init[0:min(num_of_results,len(results_init)-1)]
    results_parsed = dataset_parsers.parse_ChatGPT_Reviews(results_truncated)
    results_with_prefix = politeness_prefix.add_prefix(results_parsed,politeness,constant_variant,word_variant)

    data_arranged = ()

    # Concatenate prefix-question-query-context
    for i in range(len(results_with_prefix['query'])):
        item  =  results_with_prefix['prefix'][i] + " " + results_with_prefix['question'][i] + "? Title: \"" + results_with_prefix['query'] + "\", Content: \"" + results_with_prefix['context'] + "\""
        data_arranged["text"].append(item)
        data_arranged["answer"].append(results_with_prefix['answer'][i])
    return data_arranged

def assemble_McDonald_Reviews(num_of_results,do_load,politeness,constant_variant,word_variant):
    results_init = dataset_loaders.Load_McDonald_Reviews(do_load)
    results_truncated = results_init[0:min(num_of_results,len(results_init)-1)]
    results_parsed = dataset_parsers.parse_McDonald_Reviews(results_truncated)
    results_with_prefix = politeness_prefix.add_prefix(results_parsed,politeness,constant_variant,word_variant)

    data_arranged = ()

    # Concatenate prefix-question-query
    for i in range(len(results_with_prefix['query'])):
        item  =  results_with_prefix['prefix'][i] + " " + results_with_prefix['question'][i] + "? \"" + results_with_prefix['query'] + "\""
        data_arranged["text"].append(item)
        data_arranged["answer"].append(results_with_prefix['answer'][i])
    return data_arranged

def assemble_Fake_and_Real_News(num_of_results,do_load,politeness,constant_variant,word_variant):
    results_init = dataset_loaders.Load_Fake_and_Real_News(do_load)
    results_truncated = results_init[0:min(num_of_results,len(results_init)-1)]
    results_parsed = dataset_parsers.parse_Fake_and_Real_News(results_truncated)
    results_with_prefix = politeness_prefix.add_prefix(results_parsed,politeness,constant_variant,word_variant)

    data_arranged = ()

    # Concatenate prefix-question-query-context
    for i in range(len(results_with_prefix['query'])):
        item  =  results_with_prefix['prefix'][i] + " " + results_with_prefix['question'][i] + ": Title: \"" + results_with_prefix['query'] + "\", Text: \"" + results_with_prefix['context'] + "\""
        data_arranged["text"].append(item)
        data_arranged["answer"].append(results_with_prefix['answer'][i])
    return data_arranged

def assemble_Boolean_String(num_of_results,min_string,max_string,probability_true,politeness,constant_variant,word_variant):
    results_init = dataset_loaders.Load_Boolean_String(min_string,max_string,probability_true,num_of_results)
    # Already set to correct length - no truncation
    results_parsed = dataset_parsers.parse_Boolean_String(results_init)
    results_with_prefix = politeness_prefix.add_prefix(results_parsed,politeness,constant_variant,word_variant)

    data_arranged = ()

    # Concatenate prefix-question-query
    for i in range(len(results_with_prefix['query'])):
        item  =  results_with_prefix['prefix'][i] + " " + results_with_prefix['question'][i] + ": \"" + results_with_prefix['query'] + "\""
        data_arranged["text"].append(item)
        data_arranged["answer"].append(results_with_prefix['answer'][i])
    return data_arranged

def assemble_English_Translation(num_of_results,do_load,to_english,other_lang,politeness,constant_variant,word_variant):
    assert other_lang == "hindi" or other_lang == "french", "Unsupported language {other_lang}"
    if other_lang == "hindi":
        results_init = dataset_loaders.Load_Hindi_English_Translation(do_load,to_english)
    else:
        results_init = dataset_loaders.Load_English_French_Translation(do_load,to_english)
    results_truncated = results_init[0:min(num_of_results,len(results_init)-1)]
    results_parsed = dataset_parsers.parse_English_Translation(results_truncated,to_english,other_lang)
    results_with_prefix = politeness_prefix.add_prefix(results_parsed,politeness,constant_variant,word_variant)

    data_arranged = ()

    # Concatenate prefix-question-query
    for i in range(len(results_with_prefix['query'])):
        item  =  results_with_prefix['prefix'][i] + " " + results_with_prefix['question'][i] + ": \"" + results_with_prefix['query'] + "\""
        data_arranged["text"].append(item)
        data_arranged["answer"].append(results_with_prefix['answer'][i])
    return data_arranged

if __name__ == "__main__":
    os.system("echo entered data_assembler as main")

    print("MathQA examples:")
    datasetQA = assemble_MathQA(5,4,True,-3,False,0)
    for row in datasetQA:
        print(row)

    print("S&E examples:")
    datasetSE = assemble_SentimentsAndEmotions(5,False,True,True,-2,False,0)
    for row in datasetSE:
        print(row)
    
    print("GPT Reviews examples:")
    datasetGPT = assemble_ChatGPT_Reviews(5,True,-1,False,0)
    for row in datasetGPT:
        print(row)
   
    print("McDonald's Store Reviews examples:")
    datasetMCD = assemble_McDonald_Reviews(5,True,1,False,0)
    for row in datasetMCD:
        print(row)
   
    print("Fake and Real News examples:")
    datasetFVR = assemble_Fake_and_Real_News(5,True,2,False,0)
    for row in datasetFVR:
        print(row)
    
    print("Boolean strings examples:")
    datasetBOOL = assemble_Boolean_String(5,7,15,0.5,3,False,0)
    for row in datasetBOOL:
        print(row)
    
    print("Hindi - English Translation examples:")
    datasetHIN_ENG = assemble_English_Translation(5,True,False,"hindi",3,True,2)
    for row in datasetHIN_ENG:
        print(row)
    
    print("English - French Translation examples:")
    datasetENG_FRE = assemble_English_Translation(5,True,True,"french",-3,True,2)
    for row in datasetENG_FRE:
        print(row)

    os.system("echo clearing loaded datasets")
    dataset_loaders.clear_All_Datasets()
    os.system("echo done clearing loaded datasets")