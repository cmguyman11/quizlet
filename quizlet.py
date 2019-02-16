"""
CS124 PA5: Quizlet // Stanford, Winter 2019
by @lcruzalb, with assistance from @jchen437
"""
import sys
import getopt
import os
import math
import operator
import random
from collections import defaultdict
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize

#############################################################################
###                    CS124 Homework 5: Quizlet!                         ###
#############################################################################

# ---------------------------------------------------------------------------
# Part 1: Synonyms                                                          #
# ----------------                                                          #
# You will implement 4 functions:                                           #
#   cosine_similarity, euclidean_distance, find_synonym, part1_written      #
# ---------------------------------------------------------------------------

def cosine_similarity(v1, v2):
    '''
    Calculates and returns the cosine similarity between vectors v1 and v2

    Arguments:
        v1, v2 (numpy vectors): vectors

    Returns:
        cosine_sim (float): the cosine similarity between v1, v2
    '''
    cosine_sim = 0
    #########################################################
    ## TODO: calculate cosine similarity between v1, v2    ##
    #########################################################

    #########################################################
    ## End TODO                                            ##
    #########################################################
    return cosine_sim   

def euclidean_distance(v1, v2):
    '''
    Calculates and returns the euclidean distance between v1 and v2

    Arguments:
        v1, v2 (numpy vectors): vectors

    Returns:
        euclidean_dist (float): the euclidean distance between v1, v2
    '''
    euclidean_dist = 0
    #########################################################
    ## TODO: calculate euclidean distance between v1, v2   ##
    #########################################################

    #########################################################
    ## End TODO                                           ##
    #########################################################
    return euclidean_dist                 

def find_synonym(word, choices, embeddings, comparison_metric):
    '''
    Answer a multiple choice synonym question! Namely, given a word w 
    and list of candidate answers, find the word that is most similar to w.
    Similarity will be determined by either euclidean distance or cosine
    similarity, depending on what is passed in as the comparison_metric.

    Arguments:
        word (string): word
        choices (list of strings): list of candidate answers
        embeddings (map): map of words (strings) to their embeddings (np vectors)
        comparison_metric (string): either 'euc_dist' or 'cosine_sim'. 
            This indicates which metric to use - either euclidean distance or cosine similarity.
            With euclidean distance, we want the word with the lowest euclidean distance.
            With cosine similarity, we want the word with the highest cosine similarity.

    Returns:
        answer (string): the word in choices most similar to the given word
    '''
    answer = None
    #########################################################
    ## TODO: find synonym                                  ##
    #########################################################

    #########################################################
    ## End TODO                                            ##
    ######################################################### 
    return answer

def part1_written():
    '''
    Finding synonyms using cosine similarity on word embeddings does fairly well!
    However, it's not perfect. In particular, you should see that it gets the last
    synonym quiz question wrong (the true answer would be positive):

    30. What is a synonym for sanguine?
        a) pessimistic
        b) unsure
        c) sad
        d) positive

    What word does it choose instead? In 1-2 sentences, explain why you think 
    it got the question wrong.
    '''
    #########################################################
    ## TODO: replace string with your answer               ##
    ######################################################### 
    answer = """
    TODO fill this in
    """
    #########################################################
    ## End TODO                                            ##
    ######################################################### 
    return answer


# ---------------------------------------------------------------------------
# Part 2: Analogies                                                         #
# -----------------                                                         #
# You will implement 1 function: find_analogy_word                          #
# ---------------------------------------------------------------------------

def find_analogy_word(a, b, aa, choices, embeddings):
    '''
    Find the word bb that completes the analogy: a:b -> aa:bb
    A classic example would be: man:king -> woman:queen

    Note: use cosine similarity as your similarity metric

    Arguments:
        a, b, aa (strings): words in the analogy described above
        choices (list): list of strings for possible answer
        embeddings (map): map of words (strings) to their embeddings (np vectors)

    Returns:
        answer: the word bb that completes the analogy
    '''
    answer = None
    #########################################################
    ## TODO: analogy                                       ##
    #########################################################

    #########################################################
    ## End TODO                                            ##
    #########################################################
    return answer


# ---------------------------------------------------------------------------
# Part 3: Sentence Similarity                                               #
# ---------------------------                                               #
# You will implement 2 functions: get_embedding and get_similarity          #
# ---------------------------------------------------------------------------

def get_embedding(s, embeddings, use_POS=False, POS_weights=None):
    '''
    Returns vector embedding for a given sentence.

    Hint:
    - to get all the words in the sentence, you can use nltk's `word_tokenize` function
        >>> list_of_words = word_tokenize(sentence_string)
    - to get part of speech tags for words in a sentence, you can use `nltk.pos_tag`
        >>> tagged_tokens = nltk.pos_tag(list_of_words)
    - you can read more here: https://www.nltk.org/book/ch05.html

    Arguments:
        s (string): sentence
        embeddings (map): map of words (strings) to their embeddings (np vectors)
        use_POS (boolean): flag indicating whether to use POS weightings when
            calculating the sentence embedding
        POS_weights (map): map of part of speech tags (strings) to their weights (floats),
            it is only to be used if the use_POS flag is true

    Returns:
        embed (np vector): vector embedding of sentence s
    '''
    embed = np.zeros(embeddings.vector_size)
    #########################################################
    ## TODO: get embedding                                 ##
    #########################################################

    #########################################################
    ## End TODO                                            ##
    #########################################################
    return embed

def get_similarity(s1, s2, embeddings, use_POS, POS_weights=None):
    '''
    Given 2 sentences and the embeddings dictionary, convert the sentences
    into sentence embeddings and return the cosine similarity between them.

    Arguments:
        s1, s2 (strings): sentences
        embeddings (map): map of words (strings) to their embeddings (np vectors)
        use_POS (boolean): flag indicating whether to use POS weightings when
            calculating the sentence embedding
        POS_weights (map): map of part of speech tags (strings) to their weights (floats),
            it is only to be used if the use_POS flag is true

    Returns:
        similarity (float): cosine similarity of the two sentence embeddings
    '''
    similarity = 0
    #########################################################
    ## TODO: compute similarity                            ##
    #########################################################

    #########################################################
    ## End TODO                                            ##
    #########################################################
    return similarity


# ---------------------------------------------------------------------------
# Part 4: Exploration                                                       #
# ---------------------------                                               #
# You will implement 2 functions: occupation_exploration & part4_written    #
# ---------------------------------------------------------------------------

def occupation_exploration(occupations, embeddings):
    '''
    Given a list of occupations, return the 5 occupations that are closest
    to 'man', and the 5 closest to 'woman', using cosine similarity between
    corresponding word embeddings as a measure of similarity.

    Arguments:
        occupations (list): list of occupations (strings)
        embeddings (map): map of words (strings) to their embeddings (np vectors)

    Returns:
        top_man_occs (list): list of 5 occupations (strings) closest to 'man'
        top_woman_occs (list): list of 5 occuptions (strings) closest to 'woman'
            note: both lists should be sorted, with the occupation with highest
                  cosine similarity first in the list
    '''
    top_man_occs = []
    top_woman_occs = []
    #########################################################
    ## TODO: get 5 occupations closest to 'man' & 'woman'  ##
    #########################################################

    #########################################################
    ## End TODO                                            ##
    #########################################################
    return top_man_occs, top_woman_occs

def part4_written():
    '''
    Take a look at what occupations you found are closest to 'man' and
    closest to 'woman'. Do you notice anything curious? In 1-2 sentences,
    describe what you find, and why you think this occurs.
    '''
    #########################################################
    ## TODO: replace string with your answer               ##
    ######################################################### 
    answer = """
    TODO fill this in
    """
    #########################################################
    ## End TODO                                            ##
    ######################################################### 
    return answer

# ------------------------- Do not modify code below --------------------------------

# Functions to run each section
def part1(embeddings, synonym_qs, print_q=False):
    '''
    Calculates accuracy on part 1
    '''
    print ('Part 1: Synonyms!')
    print ('-----------------')

    acc_euc_dist = get_synonym_acc('euc_dist', embeddings, synonym_qs, print_q)
    acc_cosine_sim = get_synonym_acc('cosine_sim', embeddings, synonym_qs, print_q)

    print ('accuracy using euclidean distance: %.5f' % acc_euc_dist)
    print ('accuracy using cosine similarity : %.5f' % acc_cosine_sim)
    
    # sanity check they answered written - this is just a heuristic
    written_ans = part1_written()
    if 'TODO' in written_ans:
        print ('Part 1 written answer contains TODO, did you answer it?')

    print (' ')
    return acc_euc_dist, acc_cosine_sim

def get_synonym_acc(comparison_metric, embeddings, synonym_qs, print_q=False):
    '''
    Helper function to compute synonym answering accuracy
    '''
    if print_q:
        metric_str = 'cosine similarity' if comparison_metric == 'cosine_sim' else 'euclidean distance'
        print ('Answering part 1 using %s as the comparison metric...' % metric_str)

    n_correct = 0
    for i, (w, choices, answer) in enumerate(synonym_qs):
        ans = find_synonym(w, choices, embeddings, comparison_metric)
        if ans == answer: n_correct += 1

        if print_q:
            print ('%d. What is a synonym for %s?' % (i+1, w))
            a, b, c, d = choices[0], choices[1], choices[2], choices[3]
            print ('    a) %s\n    b) %s\n    c) %s\n    d) %s' % (a, b, c, d))
            print ('you answered: %s \n' % ans)

    acc = n_correct / len(synonym_qs)
    return acc

def part2(embeddings, analogy_qs, print_q=False):
    '''
    Calculates accuracy on part 2.
    '''
    print ('Part 2: Analogies!')
    print ('------------------')

    n_correct = 0
    for i, (tup, choices) in enumerate(analogy_qs):
        a, b, aa, true_bb = tup
        ans = find_analogy_word(a, b, aa, choices, embeddings)
        if ans == true_bb: n_correct += 1

        if print_q:
            print ('%d. %s is to %s as %s is to ___?' % (i+1, a, b, aa))
            print ('    a) %s\n    b) %s\n    c) %s\n    d) %s' % tuple(choices))
            print ('You answered: %s\n' % ans)

    acc = n_correct / len(analogy_qs)
    print ('accuracy: %.5f' % acc)
    print (' ')
    return acc

def part3(embeddings, sentence_sim_qs, print_q=False):
    '''
    Calculates accuracy of part 3.
    '''
    print ('Part 3: Sentence similarity!')
    print ('----------------------------')

    acc_base = get_sentence_sim_accuracy(embeddings, sentence_sim_qs, use_POS=False, print_q=print_q)
    acc_POS = get_sentence_sim_accuracy(embeddings, sentence_sim_qs, use_POS=True, print_q=print_q)

    print ('accuracy (regular): %.5f' % acc_base)
    print ('accuracy with POS weighting: %.5f' % acc_POS)
    print (' ')
    return acc_base, acc_POS

def get_sentence_sim_accuracy(embeddings, sentence_sim_qs, use_POS, print_q=False):
    '''
    Helper function to compute sentence similarity classification accuracy.
    '''
    THRESHOLD = 0.95
    POS_weights = load_pos_weights_map() if use_POS else None

    if print_q:
        type_str = 'with POS weighting' if use_POS else 'regular'
        print ('Answering part 3 (%s)...' % type_str)

    n_correct = 0
    for i, (label, s1, s2) in enumerate(sentence_sim_qs):
        sim = get_similarity(s1, s2, embeddings, use_POS, POS_weights)
        pred = 1 if sim > THRESHOLD else 0
        if pred == label: n_correct += 1

        if print_q:
            print ('%d. True/False: the following two sentences are semantically similar:' % (i+1))
            print ('     1. %s' % s1)
            print ('     2. %s' % s2)
            print ('You answered: %r\n' % (True if pred == 1 else False))

    acc = n_correct / len(sentence_sim_qs)
    return acc

def part4(embeddings):
    '''
    Runs part 4 functions
    '''
    print ('Part 4: Exploration!')
    print ('--------------------')

    occupations = load_occupations_list()
    top_man_occs, top_woman_occs = occupation_exploration(occupations, embeddings)
    
    print ('occupations closest to "man" - you answered:')
    for i, occ in enumerate(top_man_occs):
        print (' %d. %s' % (i+1, occ))
    print ('occupations closest to "woman" - you answered:')
    for i, occ in enumerate(top_woman_occs):
        print (' %d. %s' % (i+1, occ))

    # sanity check they answered written - this is just a heuristic
    written_ans = part4_written()
    if 'TODO' in written_ans:
        print ('Part 4 written answer contains TODO, did you answer it?')
    print (' ')
    return top_man_occs, top_woman_occs


# Helper functions to load questions
def load_synonym_qs(filename):
    '''
    input line:
        word    c1,c2,c3,c4     answer

    returns list of tuples, each of the form:
        (word, [c1, c2, c3, c4], answer)
    '''
    synonym_qs = []
    with open(filename) as f:
        f.readline()    # skip header
        for line in f:
            word, choices_str, ans = line.strip().split('\t')
            choices = [c.strip() for c in choices_str.split(',')]
            synonym_qs.append((word.strip(), choices, ans.strip()))
    return synonym_qs

def load_analogy_qs(filename):
    '''
    input line:
        a,b,aa,bb   c1,c2,c3,c4

    returns list of tuples, each of the form:
        (a, b, aa, bb)  // for analogy a:b --> aa:bb
    '''
    analogy_qs = []
    with open(filename) as f:
        f.readline()    # skip header
        for line in f:
            toks, choices_str = line.strip().split('\t')
            analogy_words = tuple(toks.strip().split(','))          # (a, b, aa, bb)
            choices = [c.strip() for c in choices_str.split(',')]   # [c1, c2, c3, c4]
            analogy_qs.append((analogy_words, choices))
    return analogy_qs

def load_sentence_sim_qs(filename):
    '''
    input line:
        label   s1  s2
    
    returns list of tuples, each of the form:
        (label, s1, s2)
    '''
    samples = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            label_str, s1, s2 = line.split('\t')
            label = int(label_str)
            samples.append((label, s1.strip(), s2.strip()))
    return samples

def load_pos_weights_map():
    '''
    Helper that loads the POS tag weights for part 3
    '''
    d = {}
    with open("data/pos_weights.txt") as f:
        for line in f:
           pos, weight = line.split()
           d[pos] = float(weight)
    return d

def load_occupations_list():
    '''
    Helper that loads the list of occupations for part 4
    '''
    occupations = []
    with open("data/occupations.txt") as f:
        for line in f:
            occupations.append(line.strip())
    return occupations

def main():
    (options, args) = getopt.getopt(sys.argv[1: ], '1234p')

    # load embeddings
    embeddings = KeyedVectors.load_word2vec_format("data/embeddings/glove50_4k.txt", binary=False)

    # load questions
    root_dir = 'data/dev/'
    synonym_qs = load_synonym_qs(root_dir + 'synonyms.csv')
    analogy_qs = load_analogy_qs(root_dir + 'analogies.csv')
    sentence_sim_qs = load_sentence_sim_qs(root_dir + 'sentences.csv')

    # if user specifies p, we'll print out the quiz questions
    PRINT_Q = False
    if ('-p', '') in options:
        PRINT_Q = True

    # if user specifies section (1-4), only run that section
    if ('-1', '') in options:
        part1(embeddings, synonym_qs, PRINT_Q)

    elif ('-2', '') in options:
        part2(embeddings, analogy_qs, PRINT_Q)

    elif ('-3', '') in options:
        part3(embeddings, sentence_sim_qs, PRINT_Q)

    elif ('-4', '') in options:
        part4(embeddings)

    # otherwise, run all 4 sections
    else:
        part1(embeddings, synonym_qs, PRINT_Q)
        part2(embeddings, analogy_qs, PRINT_Q)
        part3(embeddings, sentence_sim_qs, PRINT_Q)
        part4(embeddings)

if __name__ == "__main__":
        main()
