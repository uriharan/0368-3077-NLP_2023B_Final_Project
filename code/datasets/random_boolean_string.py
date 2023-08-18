import numpy
import math

def evaluate_gate(gate, values):
    if gate == 0: # NOT
        return not values[0]
    if gate == 1: # OR
        return values[0] or values[1]
    if gate == 2: # AND
        return values[0] and values[1]
    return (values[0] and (not values[1])) or ((not values[0]) and values[1]) #XOR

def get_random_boolean_with_probability(string_complexity, probability_true):
    """
    Inputs: int string_complexity, double probability_true
    Outputs: string boolean_string, boolean boolean_result.
    This function randomly generates a boolean expression string, with number of words (TRUE, FALSE, and gates) equal to 'string_complexity'.
    The string's chance to evaluate as TRUE is equal to the variable 'probability_true'
    The string is made by recursively seperating it into (Substring A) LOGIC_GATE (Substring B), or NOT (Substring A).
    The logic gate is chosen uniformly among NOT, OR, NOR, AND, NAND, XOR, and XNOR (can also be written as XAND).
    """
    assert probability_true >= 0 and probability_true <= 1, "probability_true not in [0,1]! {}".format(probability_true)
    assert string_complexity >= 1, "string_complexity less then 1! {}".format(string_complexity)

    boolean_result = 1 if (probability_true > numpy.random.random()) else 0 # returns 1 with probability 'probability_true'

    if string_complexity == 1: # recursion ends - truth value
        return "TRUE" if boolean_result else "FALSE", boolean_result

    if string_complexity == 2: # recursion ends - not truth value
        return "NOT (FALSE)" if boolean_result else "NOT (TRUE)", boolean_result
    
    gate_randomized = math.ceil(numpy.random.randint(0,7)/2)  # NOT, OR, AND, XOR (NOR has half the weight)

    if gate_randomized == 0: # NOT
        substring, _ = get_random_boolean_with_probability(string_complexity-1, not boolean_result)
        return "NOT ("+ substring + ")", boolean_result
    
    input_values = (1 if (0.5 > numpy.random.random()) else 0), (1 if (0.5 > numpy.random.random()) else 0) # Assign random variables to both gate inputs
    
    gate_not = evaluate_gate(gate_randomized, input_values) != boolean_result
    
    substring_A_length = numpy.random.randint(1,string_complexity-1) # substring length chosen uniformly from 1 (truth value) to complexity-2 (B is truth value)
    substring_A, _ = get_random_boolean_with_probability(substring_A_length, input_values[0])
    substring_B, _ = get_random_boolean_with_probability(string_complexity-1-substring_A_length, input_values[1])

    gate_string = ""
    if gate_randomized == 1: # OR/NOR
        gate_string = "NOR" if gate_not else "OR"
    elif gate_randomized == 2: # AND/NAND
        gate_string = "NAND" if gate_not else "AND"
    else: # == 3, XOR/XAND
        if not gate_not:
            gate_string = "XOR"
        else: # randomly choose between XNOR or XAND with equal probability
            gate_string = "XNOR" if (0.5 > numpy.random.random()) else "XAND"

    boolean_string = "(" + substring_A + ") " + gate_string + " (" + substring_B + ")"

    return boolean_string, boolean_result


if __name__ == "__main__":
    for i in range(1,20):
        print(get_random_boolean_with_probability(i,0.3))