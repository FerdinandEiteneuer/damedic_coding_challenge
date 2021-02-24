import csv
import os

def parse_csv(datapath, filename, skip_noninformative_icds=False):
    '''
    Reads in the input files "test.csv" and "train.csv".

    If skip_noninformative_icds is True, any patient case
    with an icd count of 0 or 1 will not be read in.
    '''
    path = os.path.join(datapath, filename)

    print('Parsing csv file:', path)

    data = {}
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        header = next(csv_reader)

        for case_id, icd in csv_reader:
            icds = icd.replace(',', ' ')

            if skip_noninformative_icds:

                # ignore empty and single icd patient cases
                if icds.count(' ') >= 1:
                    data[case_id] = icds

            else:
                data[case_id] = icds
    return data


def write_recommendations(patients, recommendations):
    '''
    Creates the final output file.
    '''
    filename = 'submission.csv'
    with open(filename, 'w') as file:

        writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_ALL)

        header = ["case_id", "icd"]
        writer.writerow(header)

        for patient, icds in zip(patients, recommendations):

            writer.writerow([patient, icds])

    print('Created file:', filename)
