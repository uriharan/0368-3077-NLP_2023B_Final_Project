# add a "prefix" to vectors of (question,query,context,answer) dictionaries of varying politeness levels.
# note: contains profanity.

import random

HELLO_VARIANT = ["Hi", "Hi there", "Hello", "Hello there", "Hey", "Hey there"]
SWEAR_VARIANT_1 = ["darn", "dang", "damn", "goddamn", "bloody", "shitty", "fucking"]
SWEAR_VARIANT_2 = ["fucker", "twat", "idiot", "ass", "a-hole", "asshole", "dickhead"]

# creates prefixes with a given politeness feature - positive for polite, negative for impolite.
# Due to asking for the model's response, it is impossible to be "politeness neutral", as the way that is conveyed to the model is inherently "polite" or "impolite".
# Politeness:
# 1. Using "Please"
# 2. Adds requesting instead of commanding
# 3. Adds using "Hello" variants
# Impoliteness:
# 1. Bare commands
# 2. Adds curtness
# 3. Replaces curtness with swear words
def add_prefix(in_vec,politeness,constant_variant,word_variant):
    assert politeness != 0, "cannot be politeness-neutral"
    assert politeness in range(-3,4), "politeness not integer from -3 to 3, {politeness}"
    if not constant_variant:
        assert word_variant >= 0 and type(word_variant) == int, "word_variant not non-negative integer, {word_variant}"

    if politeness == 3:
        assert word_variant < len(HELLO_VARIANT), "word variant in politeness prefix out of scope! {politeness},{word_variant}"
    if politeness == -3:
        assert word_variant < len(SWEAR_VARIANT_1) + len(SWEAR_VARIANT_2) , "word variant in politeness prefix out of scope! {politeness},{word_variant}"\
    
    for item in in_vec:
        if(politeness == 1): # using "please"
            item["prefix"] = "Please answer the following:"
        if(politeness == 2): # using "please" and requesting
            item["prefix"] = "Could you please answer the following:"
        if(politeness == 3): # using "please", "hello", and requesting
            if not constant_variant:
                word_variant = random.randint(0,len(HELLO_VARIANT)-1)
            item["prefix"] = HELLO_VARIANT[word_variant] + ", Could you please answer the following:"
        if(politeness == -1): # using bare command
            item["prefix"] = "Answer the following:"
        if(politeness == -2): # using bare command, being curt
            item["prefix"] = "Answer:"
        if(politeness == -3): # using bare command, using swear words
            if not constant_variant:
                word_variant = random.randint(0,len(SWEAR_VARIANT_1)+len(SWEAR_VARIANT_2)-1)
            if(word_variant < len(SWEAR_VARIANT_1)):
                item["prefix"] = "Answer the " + SWEAR_VARIANT_1[word_variant] + " following:"
            else:
                item["prefix"] = "Answer the following, " + SWEAR_VARIANT_2[word_variant-len(SWEAR_VARIANT_1)] + ":"
    return in_vec
