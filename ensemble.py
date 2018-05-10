import argparse
import numpy as np
import random
from os.path import join
from pathlib import Path
import itertools
import re
import os
import sys
import jpeg4py as jpeg
import math
import csv
from collections import defaultdict

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

def most_common(lst):
    return max(set(lst), key=lst.count)

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

# TODO tf seed

parser = argparse.ArgumentParser()
# general
parser.add_argument('-c', '--csv-files', type=str, nargs='+', help='Epoch to run')
parser.add_argument('-e', '--ensemble-csv', default='ensemble.csv', help='Ensemble output filename')
args = parser.parse_args()
preds = defaultdict(list)
missing = set()
n_csvs = len(args.csv_files)
max_scores = [0.] * n_csvs

for i, csv_file in enumerate(args.csv_files):
    print("Reading {}".format(csv_file))
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader)
        for row in reader:
            idx, prediction = row[0], row[1].split(' ')
            if len(prediction) == 2:
                landmark, score = prediction
                landmark, score = int(landmark), float(score)
                max_scores[i] = max(max_scores[i], score)
                preds[idx].append((landmark, score))
            else:
                preds[idx].append((None, 0.))
                missing.add(idx)              

ensemble = { }
agreements = [0] * n_csvs

distractors = missing_rows = rows = 0     
with open(args.ensemble_csv, 'w') as csvfile:

    csv_writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['id','landmarks'])

    for idx, predictions in preds.items():
        landmarks, scores = zip(*predictions)
        # normalize scores between 0-1
        scores = [score/max_score for score,max_score in zip(scores, max_scores)]
        # remove predictions with no landmark
        predictions = [(landmarks[i], scores[i]) for i,_landmark in enumerate(landmarks) if _landmark != None ]
        if predictions == []:
            continue
        landmarks, scores = zip(*predictions)
        
        landmark = most_common(landmarks)
        landmark_scores   = [scores[i] for i,_landmark in enumerate(landmarks) if _landmark == landmark ]
        n_agreements = landmarks.count(landmark)
        agreements[n_csvs - n_agreements] += 1 
        if n_agreements >= 2:
            # agreement same
            score    = np.mean(landmark_scores) + n_agreements
        else:
            # all different
            score    = max(scores)
            landmark = landmarks[argmax(scores)]

        if (landmark != 15000) and (landmark != -1):
            ensemble[idx] = (landmark, score)
            csv_writer.writerow([idx, "{} {}".format(landmark, score)])
            rows += 1
        else:
            distractors += 1
            missing.add(idx)              

    # missing test items
    for idx in missing.difference(ensemble.keys()):
        csv_writer.writerow([idx, ""])
        rows += 1
        missing_rows += 1

voting_summary = ("Agreements: All/all except 1/all except 2/.../no agreements/missing/distractors: " + \
    '/'.join(['{}'] * (n_csvs + 2)) + ' ' + '/'.join(['{:.2f}%'] * (n_csvs+2))).format(
    *agreements, missing_rows, distractors,
    *map(lambda x: 100.*x/rows, agreements), 100. * missing_rows/rows, 100. * distractors/rows)

print(voting_summary)

print("kaggle competitions submit -f {} -m '{} {}'".format(
    args.ensemble_csv,
    voting_summary, ' '.join(sys.argv)
    ))
        