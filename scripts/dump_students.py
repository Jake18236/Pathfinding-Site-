#!/usr/bin/env python3

import csv

for student in csv.DictReader(open('static/csv/students.csv')):
    print(student['SIS User ID'])
