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
max_scores = [0.] * len(args.csv_files)

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

print(max_scores)
ensemble = { }
all_matches = some_matches = no_matches = 0
   
rows = 0     
with open(args.ensemble_csv, 'w') as csvfile:

    csv_writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['id','landmarks'])

    for idx, predictions in preds.items():
        landmarks, scores = zip(*predictions)
        scores = [score/max_score for score,max_score in zip(scores, max_scores)]
        predictions = [(landmarks[i], scores[i]) for i,_landmark in enumerate(landmarks) if _landmark != None ]
        landmarks, scores = zip(*predictions)

        if landmarks.count(landmarks[0]) == len(landmarks):
            # all same
            landmark = landmarks[0]
            score    = max(scores)
            if len(landmarks) == len(args.csv_files):
                score += 2.
                all_matches += 1
            else:
                score += 2.
        elif len(set(landmarks)) == len(landmarks):
            # all different
            no_matches += 1
            score    = max(scores)
            landmark = landmarks[argmax(scores)]
        else:
            # some matches
            some_matches += 1
            landmark = most_common(landmarks)
            scores   = [scores[i] for i,_landmark in enumerate(landmarks) if _landmark == landmark ]
            score    = max(scores) + 1.
        ensemble[idx] = (landmark, score)
        csv_writer.writerow([idx, "{} {}".format(landmark, score)])
        rows += 1
    # missing test items
    for idx in missing.difference(ensemble.keys()):
        csv_writer.writerow([idx, ""])
        rows += 1


print("Full/partial/no matches: {}/{}/{} {:.2f}%/{:.2f}%/{:.2f}%".format(
    all_matches, some_matches, no_matches,
    100. * all_matches/rows, 100. * some_matches / rows, 100. * no_matches /rows))
    #csv_writer.writerow([idx, ""])

print("kaggle competitions submit -f {} -m '{}'".format(
    args.ensemble_csv,
    ' '.join(sys.argv)
    ))
        