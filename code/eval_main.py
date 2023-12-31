# This is the file for use in evaluation of the various models.
# It recieves the parameters for which datasets and which models to run on, at which politeness level, and performs an evaluation run on them.

import os
import dataset_loaders # for clearing datasets
import data_assembler # Get queries
import model_users # Get models
import pandas as pd # Save data
import argparse # Parse argument



OUTPUT_FOLDER = "./generated/outputs"

# Models supported
MODEL_NAMES = ["flan-t5-base", "flan-t5-small", "falcon-7b-instruct", "falcon-40b-instruct","fastchat-t5"]
MODEL_LOADERS = [model_users.Load_FlanT5base, model_users.Load_FlanT5small, model_users.Load_Falcon7BInstruct, model_users.Load_Falcon40BInstruct,model_users.Load_FastchatT5]
MODEL_MAXLENGTH = [512,512,1024,1024,512]
# Datasets supported
DATASET_NAMES = ["mathqa", "sentiments-type", "emotions-type", "chatgpt-review", "mcdonald-review","fake-v-real_news","hi-en","en-hi","en-fr","fr-en","boolean-string"]

# Evaluation function - currently string equality.
# Returns number between 0 and the input length, corresponding to the returned score.
def calc_correctness(gold_answers,model_answers):
    correct = 0
    for index in range(len(gold_answers)):
        gold_answer = gold_answers[index]
        model_answer = model_answers[index]
        if (gold_answer == model_answer):
            correct += 1
    return correct

# Recieves a dataset, model, and tokenizer. Evaluates the model performance on the dataset.
# Dataset format: Dictionary with keys:
#   "text": Vector of questions.
#   "answer": Vector of gold responses.
# model/tokenizer are from Huggingface's Transformers package.
# Evaluation is string equality between the golden answers and the model answers.
# To ensure no Uppercase/Lowercase conflicts, all answers are transformed to lowercase.
# Can be improved - does not support partial answers, chain-of-though, or non-string results, like closeness to a number.
def eval_dataset_on_model(dataset,model,tokenizer,save_results,identifier):
    print("Starting Run! Number of examples: " + str(len(dataset["text"])))
    outputs = model_users.Run_Model(model,tokenizer,dataset["text"])
    print("Run Complete! First result example: " + str(outputs[0]))

    correct = calc_correctness(dataset["answer"],outputs)
    
    print("Result: " + str(correct) + " correct out of " + str(len(outputs)) + ", " + str(100*correct/len(outputs)) + "% success rate")

    if save_results:
        location = OUTPUT_FOLDER + "/" + identifier + ".csv"
        print("Saving results to Disk! location " + location)
        # Organize data into one struct
        data = {
            "query": dataset["text"],
            "gold_answer": dataset["answer"],
            "model_answer": outputs
        }
        # Load data into a DataFrame object:
        df = pd.DataFrame(data)
        # Ensure directory exists
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER) 
        # Store on disk
        df.to_csv(location)

    return correct, len(outputs)

def run_eval(dataset_name,model_name,politeness,save_results,num_results_max,min_index):
    assert model_name in MODEL_NAMES, "model not found! " + model_name
    assert dataset_name in DATASET_NAMES, "dataset not found! " + dataset_name

    print("Called run_eval! running on model " + model_name + ", dataset " + dataset_name + " from " + str(min_index) + ", politeness " + str(politeness))

    model, tokenizer = MODEL_LOADERS[MODEL_NAMES.index(model_name)]()
    max_length = MODEL_MAXLENGTH[MODEL_NAMES.index(model_name)]

    if dataset_name == "mathqa":
        dataset = data_assembler.assemble_MathQA(num_of_results=num_results_max,set_num=7,do_load=True,politeness=politeness,constant_variant=False,word_variant=0,min_index=min_index)
    if dataset_name == "sentiments-type":
        dataset = data_assembler.assemble_SentimentsAndEmotions(num_of_results=num_results_max,to_emotion=False,do_load=True,variant_score=False,politeness=politeness,constant_variant=False,word_variant=0,min_index=min_index)
    if dataset_name == "emotions-type":
        dataset = data_assembler.assemble_SentimentsAndEmotions(num_of_results=num_results_max,to_emotion=True,do_load=True,variant_score=False,politeness=politeness,constant_variant=False,word_variant=0,min_index=min_index)
    if dataset_name == "mcdonald-review":
        dataset = data_assembler.assemble_McDonald_Reviews(num_of_results=num_results_max,do_load=True,politeness=politeness,constant_variant=False,word_variant=0,min_index=min_index)
    if dataset_name == "chatgpt-review":
        dataset = data_assembler.assemble_ChatGPT_Reviews(num_of_results=num_results_max,do_load=True,politeness=politeness,constant_variant=False,word_variant=0,min_index=min_index)
    if dataset_name == "fake-v-real_news":
        dataset = data_assembler.assemble_Fake_and_Real_News(num_of_results=num_results_max,do_load=True,politeness=politeness,constant_variant=False,word_variant=0,min_index=min_index)
    if dataset_name == "hi-en":
        dataset = data_assembler.assemble_English_Translation(num_of_results=num_results_max,do_load=True,to_english=True,other_lang="hindi",politeness=politeness,constant_variant=False,word_variant=0,min_index=min_index)
    if dataset_name == "en-hi":
        dataset = data_assembler.assemble_English_Translation(num_of_results=num_results_max,do_load=True,to_english=False,other_lang="hindi",politeness=politeness,constant_variant=False,word_variant=0,min_index=min_index)
    if dataset_name == "fr-en":
        dataset = data_assembler.assemble_English_Translation(num_of_results=num_results_max,do_load=True,to_english=True,other_lang="french",politeness=politeness,constant_variant=False,word_variant=0,min_index=min_index)
    if dataset_name == "en-fr":
        dataset = data_assembler.assemble_English_Translation(num_of_results=num_results_max,do_load=True,to_english=False,other_lang="french",politeness=politeness,constant_variant=False,word_variant=0,min_index=min_index)
    if dataset_name == "boolean-string":
        dataset = data_assembler.assemble_Boolean_String(num_of_results=num_results_max,min_string=4,max_string=10,probability_true=0.5,politeness=politeness,constant_variant=False,word_variant=0)

    dataset_truncated = {"text":[], "answer":[]} # Dataset of only short-enough strings
    for index in range(len(dataset["text"])):
        if len(dataset["text"][index]) <= max_length:
            dataset_truncated["text"].append(dataset["text"][index])
            dataset_truncated["answer"].append(dataset["answer"][index])

    print("Dataset truncated to max model length: remaining " + str(len(dataset_truncated["text"])) + " lines out of " + str(len(dataset["text"])))

    identifier = "table_" + model_name + "_running_" + dataset_name + "_politeness_" + str(politeness) +"_from_" + str(min_index) + "_to_" + str(min_index+num_results_max-1)

    eval_dataset_on_model(dataset=dataset_truncated,model=model,tokenizer=tokenizer,save_results=save_results,identifier=identifier)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help="name of the dataset evaluated on", type=str)
    parser.add_argument("model_name", help="name of the model being evaluated", type=str)
    parser.add_argument("num_results_max", help="uses rows up to here only from the dataset, if the dataset is smaller use all rows", type=int)
    parser.add_argument("--save_results", help="whether to save the results to disk", action="store_true")
    parser.add_argument("--allpoliteness", help="use every politeness level", action="store_true")
    parser.add_argument("--politeness", help="level of politeness to be used. allowed values: -3,-2,-1,1,2,3", type=int)
    parser.add_argument("--min_index", help="start reading dataset from this index", type=int, default=0)
    args = parser.parse_args()

    os.system("echo entered eval_main as main")
    assert args.allpoliteness or (args.politeness != None), "Must have a politeness option!"
    if(args.allpoliteness):
        run_eval(dataset_name=args.dataset_name,model_name=args.model_name,politeness=-3,save_results=args.save_results,num_results_max=args.num_results_max,min_index=args.min_index)
        run_eval(dataset_name=args.dataset_name,model_name=args.model_name,politeness=-2,save_results=args.save_results,num_results_max=args.num_results_max,min_index=args.min_index)
        run_eval(dataset_name=args.dataset_name,model_name=args.model_name,politeness=-1,save_results=args.save_results,num_results_max=args.num_results_max,min_index=args.min_index)
        run_eval(dataset_name=args.dataset_name,model_name=args.model_name,politeness=1,save_results=args.save_results,num_results_max=args.num_results_max,min_index=args.min_index)
        run_eval(dataset_name=args.dataset_name,model_name=args.model_name,politeness=2,save_results=args.save_results,num_results_max=args.num_results_max,min_index=args.min_index)
        run_eval(dataset_name=args.dataset_name,model_name=args.model_name,politeness=3,save_results=args.save_results,num_results_max=args.num_results_max,min_index=args.min_index)
    else:
        run_eval(dataset_name=args.dataset_name,model_name=args.model_name,politeness=args.politeness,save_results=args.save_results,num_results_max=args.num_results_max,min_index=args.min_index)

    # Clean folders
    dataset_loaders.clear_All_Datasets()
    model_users.clear_All_Models()