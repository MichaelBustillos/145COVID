# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import csv

x_axis = {}
inf_y = {}
death_y = {}

with open("/kaggle/input/2021spring-cs145-ucla-covid19-prediction/train_trendency.csv", "r") as csv_file:
    reader = csv.reader(csv_file)
    
    first_row = True
    for row in reader:
        if first_row:
            first_row = False
            continue
            
        state = row[1]
        x_axis[state] = []
        inf_y[state] = []
        death_y[state] = []
        
        if state == "Wyoming":
            break

import datetime

with open("/kaggle/input/2021spring-cs145-ucla-covid19-prediction/train_trendency.csv", "r") as csv_file:
    reader = csv.reader(csv_file)

    first_row = True
    for row in reader:
        if first_row:
            first_row = False
            continue
       
        state = row[1]
        
        date_string = row[2]
        mdy = date_string.split("-")
        date = datetime.date(int(mdy[2]), int(mdy[0]), int(mdy[1]))
        
        inf = int(row[3])
        deaths = int(row[4])
        
        x_axis[state].append(date)
        inf_y[state].append(inf)
        death_y[state].append(deaths)

import matplotlib.dates as mdates

coeff_inf = {}
coeff_death = {}
for state in x_axis:
    coeff_inf[state] = []
    coeff_death[state] = []

for state in x_axis:
    x = mdates.date2num(x_axis[state])
    x = x - 18638
    log_x = np.log(x)
    c_inf = np.polyfit(log_x, inf_y[state], 1)
    c_death = np.polyfit(log_x, death_y[state], 1)
    
    coeff_inf[state].append(c_inf[0])
    coeff_inf[state].append(c_inf[1])
    
    coeff_death[state].append(c_death[0])
    coeff_death[state].append(c_death[1])

start = 80
out = "ID,Confirmed,Deaths\n"

id = 0
for x in range(start, start+30):
    for state in x_axis:
        predicted_inf = coeff_inf[state][0]*np.log(x) + coeff_inf[state][1]
        predicted_death = coeff_death[state][0]*np.log(x) + coeff_death[state][1]
        
        out += str(id) + "," + str(predicted_inf) + "," + str(predicted_death) + "\n"
        
        id = id + 1
            
f = open("/kaggle/working/submission.csv", "w")
f.write(out)
