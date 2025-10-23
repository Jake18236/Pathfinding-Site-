#!/usr/bin/env python3

import csv
import random
import glob
import sys
import yaml

# Load TAs from semester_info.yaml
semester_info = yaml.safe_load(open('../static/yaml/semester_info.yaml'))
TAS = [ta['github'] for ta in semester_info['TAs'].values()]

TEAMS = set()  # Use a set to store unique team names

for student in csv.DictReader(open('../static/csv/teams.csv', 'r')):
    TEAMS.add(student['group_name'])  # Add team names to the set

TEAMS = list(TEAMS)  # Convert set back to list for processing

for i in range(1, 6):
    # Update conflicts to use semester_info.yaml
    CONFLICTS = dict(
        (ta['github'], ta.get('conflicts', []))
        for ta in semester_info['TAs'].values()
    )

    HAS_CONFLICTS = True

    while HAS_CONFLICTS:
        random.shuffle(TEAMS)
        random.shuffle(TAS)

        TAS           = TAS * (len(TEAMS) // len(TAS) + 1)
        MAPPING       = list(sorted(map(list, zip(TEAMS, TAS))))
        HAS_CONFLICTS = False

        for ta, conflicts in CONFLICTS.items():
            for conflict in conflicts:
                if [conflict, ta] in MAPPING:
                    HAS_CONFLICTS = True

    # Write each mapping to a separate file
    filename = f'../static/yaml/homework{i:02}_teams_tas_mapping.yaml'
    with open(filename, 'w') as file:
        yaml.dump(MAPPING, file, default_flow_style=False)