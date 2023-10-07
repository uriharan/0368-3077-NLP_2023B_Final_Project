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
    model = AutoModelLib.from_pretrained(name_from_pretrained)
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
    batch = tokenizer(input)
    tensor_batch = torch.tensor(batch)
    outputs = model(**tensor_batch)
    return outputs