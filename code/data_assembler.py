# This file includes code designed to combine the dataset_loaders.py, dataset_parsers.py, and politeness_prefix.py into huggingface-compatible examples and answers.
# First - load the query-context-answer tuple of vectors, and truncate it to the wanted length.
# Second - add the prefix and dataset-based question to get a prefix-question-query-context-answer vector tuple.
# Third - combine the first several vectors, get text-answer

import dataset_loaders
import dataset_parsers
import politeness_prefix

def assemble_MathQA(num_of_results,set_num,do_load,politeness,constant_variant,word_variant):
    results_init = dataset_loaders.Load_MathQA(set_num,do_load)
    results_truncated = results_init[0:min(num_of_results,len(results_init)-1)]
    results_parsed = dataset_parsers.parse_MathQA(results_truncated)
    results_with_prefix = politeness_prefix.add_prefix(results_parsed,politeness,constant_variant,word_variant)
    
    data_arranged = ()

    for i in range(len(results_with_prefix['query'])):
        item = results_with_prefix['prefix'][i] + " " + results_with_prefix['question'][i] + ": " + results_with_prefix['query'] + " " + results_with_prefix['context'] + "."
        data_arranged["text"].append(item)
        data_arranged["answer"].append(results_with_prefix['answer'][i])
    
    return data_arranged

def assemble_SentimentsAndEmotions(num_of_results,to_emotion,do_load,variant_score,politeness,constant_variant,word_variant):
    results_init = dataset_loaders.Load_SentimentsAndEmotions(to_emotion,do_load,variant_score)
    results_truncated = results_init[0:min(num_of_results,len(results_init)-1)]
    results_parsed = dataset_parsers.parse_SentimentsAndEmotions(results_truncated,to_emotion,variant_score)
    results_with_prefix = politeness_prefix.add_prefix(results_parsed,politeness,constant_variant,word_variant)
    
    data_arranged = ()

    for i in range(len(results_with_prefix['query'])):
        if not variant_score:
            item = results_with_prefix['prefix'][i] + " " + results_with_prefix['question'][i] + "? \"" + results_with_prefix['query'] + "\""
        else:
            item = results_with_prefix['prefix'][i] + " " + results_with_prefix['question'][i] + "? Review: \"" + results_with_prefix['query'] + "\", Estimation: \"" + results_with_prefix['context'] + "\""
        data_arranged["text"].append(item)
        data_arranged["answer"].append(results_with_prefix['answer'][i])
    
    return data_arranged

    