import csv
import os
import sys
# from sklearn.metrics.pairwise import cosine_similarity
from statistics import mean

import numpy as np


## Variation of Pearson correlation calculated using the standard deviation from zero rather than from the mean value. 
def uncentered_pearson(x, y):
    # find the lengths of the x and y vectors
    x_length = len(x)
    y_length = len(y)

    # check to see if the vectors have the same length
    if x_length is not y_length:
        sys.exit('The vectors that you entered are not the same length')
        return False

    # calculate the numerator and denominator
    xy = 0
    xx = 0
    yy = 0
    for i in range(x_length):
        xy = xy + x[i] * y[i]
        xx = xx + x[i] ** 2.0
        yy = yy + y[i] ** 2.0

    # calculate the uncentered pearsons correlation coefficient
    uxy = xy / np.sqrt(xx * yy)

    return uxy


#### MAIN ####
def evaluate1(gold_folder, submission_folder):
    # input_dir = os.getcwd()

    scores = []

    languages = ('en', 'hr', 'sl')
    accepted_files = [f'results_subtask1_{lan}.tsv' for lan in languages]

    # submission_dir = os.path.join(submission_folder, 'submission_subtask1')
    # print('abc')
    print(submission_folder)
    submission_dir = submission_folder
    submitted_files = os.listdir(submission_dir)

    # ref_dir = os.path.join(gold_folder, 'gold')
    ref_dir = gold_folder

    ## Checking submission is not empty
    if len(os.listdir(submission_dir)) == 0:
        sys.exit(
            'Not able to find result files.\nPlease make sure the submitted zip file contains the right result files.')

    # Checking all files have the proper names
    for file in submitted_files:
        if file not in accepted_files:
            accepted_string = '\n\t- ' + '\n\t- '.join(accepted_files)
            sys.exit(f"Submitted file '{file}' is not a valid result file:{accepted_string}")

    for lan in languages:

        ## Looking for gold reference file
        gold_file_name = os.path.join(ref_dir, f'gold_{lan}.tsv')

        if not os.path.exists(gold_file_name):
            message = 'Couldn\'t find the expected gold scores file: {0}'
            sys.exit(message.format(gold_file_name))

        ## Looking for submission file or baseline for this language
        submission_file_name = os.path.join(submission_dir, f'results_subtask1_{lan}.tsv')
        if not os.path.exists(submission_file_name):
            submission_file_name = os.path.join(ref_dir, f'baseline_subtask1_{lan}.tsv')

        ## Loading gold reference results
        gold = []
        with open(gold_file_name, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            next(csvreader)
            for row in csvreader:
                gold.append(float(row[2]))

        ## Loading submission/baseline results
        results = []
        with open(submission_file_name, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)

            ## Checking the header and the number of columns
            row = next(csvreader)
            if (row[0] != 'change'):
                sys.exit(f"File results_subtask1_{lan}.tsv: Wrong format or missing header (should be 'diff')")
            if (len(row) != 1):
                sys.exit(f'File results_subtask1_{lan}.tsv: Wrong number of columns (should be 1)')
            for row in csvreader:
                results.append(float(row[0]))

        ## Checking number of rows
        if (len(results) != len(gold)):
            sys.exit(f'File results_subtask1_{lan}.tsv: Wrong number of values (should be {len(gold)})')

        u_pearson = uncentered_pearson(results, gold)
        scores.append(u_pearson)

    print('SUBTASK1 SCORING:')
    print(f'score:{mean(scores)}')
    print(f'english:{scores[0]}')
    print(f'estonian:{0}')
    print(f'finnish:{0}')
    print(f'croatian:{scores[1]}')
    print(f'slovenian:{scores[2]}\n')
