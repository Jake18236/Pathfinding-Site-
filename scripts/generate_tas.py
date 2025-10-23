#!/usr/bin/env python3

import csv
import random
import glob
import sys
import yaml

TAS      = [ta['github'] for ta in yaml.safe_load(open('../static/yaml/tas.yaml'))]
STUDENTS = []

for student in csv.DictReader(open('../static/csv/students.csv', 'r')):
    STUDENTS.append(student['SIS User ID'])

for i in range(0, 12):
    CONFLICTS = dict(
        (ta['github'], ta['conflicts'])
        for ta in yaml.safe_load(open('../static/yaml/tas.yaml'))
        if ta.get('conflicts')
    )

    HAS_CONFLICTS = True

    while HAS_CONFLICTS:
        random.shuffle(STUDENTS)
        random.shuffle(TAS)

        TAS           = TAS * (len(STUDENTS) // len(TAS) + 1)
        MAPPING       = list(sorted(map(list, zip(STUDENTS, TAS))))
        HAS_CONFLICTS = False

        for ta, conflicts in CONFLICTS.items():
            for conflict in conflicts:
                if [conflict, ta] in MAPPING:
                    HAS_CONFLICTS = True

    # Write each mapping to a separate file
    filename = f'../static/yaml/homework{i:02}.yaml'
    with open(filename, 'w') as file:
        yaml.dump(MAPPING, file, default_flow_style=False)