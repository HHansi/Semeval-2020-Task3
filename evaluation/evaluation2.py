import csv
import os
import sys
from statistics import mean, harmonic_mean

from scipy.stats import spearmanr, pearsonr


def evaluate2(gold_folder, submission_folder):
    # input_dir = os.getcwd()

    scores = []

    languages = ('en', 'hr', 'sl')
    accepted_files = [f'results_subtask2_{lan}.tsv' for lan in languages]

    # submission_dir = os.path.join(input_dir, 'submission_subtask2')
    submission_dir = submission_folder
    submitted_files = os.listdir(submission_dir)

    # ref_dir = os.path.join(input_dir, 'gold')
    ref_dir = gold_folder

    ## Checking submission is not empty
    if len(os.listdir(submission_dir)) == 0:
        sys.exit(
            'Not able to find result files.\nPlease make sure the submitted zip file contains the right result files.')

    ## Checking all files have the proper names
    for file in submitted_files:
        if file not in accepted_files:
            accepted_string = '\n\t- ' + '\n\t- '.join(accepted_files)
            sys.exit(f"Submitted file '{file}' is not a valid result file:{accepted_string}")

    for lan in languages:

        ## Looking for gold standard file
        gold_file_name = os.path.join(ref_dir, f'gold_{lan}.tsv')

        if not os.path.exists(gold_file_name):
            message = 'Couldn\'t find the expected gold scores file: {0}'
            sys.exit(message.format(gold_file_name))

        ## Looking for submission file or baseline for this language
        submission_file_name = os.path.join(submission_dir, f'results_subtask2_{lan}.tsv')
        if not os.path.exists(submission_file_name):
            submission_file_name = os.path.join(ref_dir, f'baseline_subtask2_{lan}.tsv')

        ## Loading gold standard results
        gold = []
        with open(gold_file_name, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            next(csvreader)
            for row in csvreader:
                gold.append(float(row[0]))
                gold.append(float(row[1]))

        ## Loading submission/baseline results
        results = []
        with open(submission_file_name, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)

            ## Checking the header and the number of columns
            row = next(csvreader)
            if (row[0] != 'sim_context1' or row[1] != 'sim_context2'):
                sys.exit(
                    f"File results_subtask2_{lan}.tsv: Wrong format or missing header (should be 'sim_context1' and 'sim_context2')")
            if (len(row) != 2):
                sys.exit(f'File results_subtask2_{lan}.tsv: Wrong number of columns (should be 2)')
            for row in csvreader:
                results.append(float(row[0]))
                results.append(float(row[1]))

        ## Checking number of rows
        if (len(results) != len(gold)):
            sys.exit(f'File results_subtask2_{lan}.tsv: Wrong number of values (should be {len(gold)})')

        pearson = pearsonr(results, gold)[0]
        spearman = spearmanr(results, gold)[0]

        score = harmonic_mean([pearson, spearman])
        scores.append(score)

        print(f'Language: {lan.upper()}')
        print(f'Pearson: {pearson}')
        print(f'Spearman: {spearman}')
        print(f'Harmonic Mean: {score}\n')

    print('SUBTASK2 SCORE:')
    print(f'score:{mean(scores)}')
    print(f'english:{scores[0]}')
    print(f'estonian:{0}')
    print(f'finnish:{0}')
    print(f'croatian:{scores[1]}')
    print(f'slovenian:{scores[2]}\n')
