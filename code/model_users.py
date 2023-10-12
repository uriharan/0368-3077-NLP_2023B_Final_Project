# This file contains commands which load models and tokenizers from huggingface.

import os
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

MODEL_LOCAL_FOLDER = "./generated/models"

FLAN_T5_BASE_REFERENCE = "google/flan-t5-base"
FLAN_T5_BASE_AUTOMODEL = AutoModelForSeq2SeqLM

FLAN_T5_SMALL_REFERENCE = "google/flan-t5-small"
FLAN_T5_SMALL_AUTOMODEL = AutoModelForSeq2SeqLM

FALCON_7B_INSTRUCT_REFERENCE = "tiiuae/falcon-7b-instruct"
FALCON_7B_INSTRUCT_AUTOMODEL = AutoModelForCausalLM

FALCON_40B_INSTRUCT_REFERENCE = "tiiuae/falcon-40b-instruct"
FALCON_40B_INSTRUCT_AUTOMODEL = AutoModelForCausalLM


# Clear all downloaded models
def clear_All_Models():
    print("Removing all downloaded models")
    os.system("rm -rf " + MODEL_LOCAL_FOLDER)

# load a generic model. uses a given AutoModel library and the generic AutoTokenizer.
def Load_Model_From_Pretrained(AutoModelLib,name_from_pretrained):
    print("Loading Model " + name_from_pretrained + " from pretrained checkpoint.")
    model = AutoModelLib.from_pretrained(name_from_pretrained, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(name_from_pretrained)
    return model, tokenizer

def Load_FlanT5base():
    return Load_Model_From_Pretrained(FLAN_T5_BASE_AUTOMODEL,FLAN_T5_BASE_REFERENCE)


def Load_FlanT5small():
    return Load_Model_From_Pretrained(FLAN_T5_SMALL_AUTOMODEL,FLAN_T5_SMALL_REFERENCE)


def Load_Falcon7BInstruct():
    return Load_Model_From_Pretrained(FALCON_7B_INSTRUCT_AUTOMODEL,FALCON_7B_INSTRUCT_REFERENCE)


def Load_Falcon40BInstruct():
    return Load_Model_From_Pretrained(FALCON_40B_INSTRUCT_AUTOMODEL,FALCON_40B_INSTRUCT_REFERENCE)

# Run a downloaded model and trainer on an array of input texts
def Run_Model(model, tokenizer, input):
    # Find the maximum sequence length in the input texts
    max_length = max(len(tokenizer.encode(text)) for text in input)
    
    # For using GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize the input strings
    encoding = tokenizer(input, padding=True, max_length=max_length, truncation=True, return_tensors="pt").to(device)

    print("Preprocessing data done! Now running model")
    
    # Run the model on the tokenized inputs
    with torch.no_grad():
        outputs = model.generate(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return outputs