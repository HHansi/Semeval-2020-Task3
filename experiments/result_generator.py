# Created by Hansi at 2/23/2020
import csv

from data_processing.file_util import create_folder_if_not_exist


def write_results(similarities, output_folder_path, language):
    columns = ['change']
    output_file_path1 = output_folder_path + '/res1/results_subtask1_' + language + '.tsv'
    create_folder_if_not_exist(output_file_path1, is_file_path=True)
    with open(output_file_path1, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='', escapechar='~')
        csvwriter.writerow(columns)
        for sim in similarities:
            change = sim[1] - sim[0]
            csvwriter.writerow([change])

    columns = ['sim_context1', 'sim_context2']
    output_file_path2 = output_folder_path + '/res2/results_subtask2_' + language + '.tsv'
    create_folder_if_not_exist(output_file_path2, is_file_path=True)
    with open(output_file_path2, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='', escapechar='~')
        csvwriter.writerow(columns)
        for sim in similarities:
            csvwriter.writerow(sim)