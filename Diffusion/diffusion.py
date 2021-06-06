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

# https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# get total population of each state
# data taken from wikipedia
pop = {}
pop["California"]=39512223
pop["Texas"]=28995881
pop["Florida"]=21477737
pop["New York"]=19453561
pop["Pennsylvania"]=12801989
pop["Illinois"]=12671821
pop["Ohio"]=11689100
pop["Georgia"]=10617423
pop["North Carolina"]=10488084
pop["Michigan"]=9986857
pop["New Jersey"]=8882190
pop["Virginia"]=8535519
pop["Washington"]=7614893
pop["Arizona"]=7278717
pop["Massachusetts"]=6949503
pop["Tennessee"]=6833174
pop["Indiana"]=6732219
pop["Missouri"]=6137428
pop["Maryland"]=6045680
pop["Wisconsin"]=5822434
pop["Colorado"]=5758736
pop["Minnesota"]=5639632
pop["South Carolina"]=5148714
pop["Alabama"]=4903185
pop["Louisiana"]=4648794
pop["Kentucky"]=4467673
pop["Oregon"]=4217737
pop["Oklahoma"]=3956971
pop["Connecticut"]=3565287
pop["Utah"]=3205958
pop["Iowa"]=3155070
pop["Nevada"]=3080156
pop["Arkansas"]=3017825
pop["Mississippi"]=2976149
pop["Kansas"]=2913314
pop["New Mexico"]=2096829
pop["Nebraska"]=1934408
pop["Idaho"]=1787065
pop["West Virginia"]=1792147
pop["Hawaii"]=1415872
pop["New Hampshire"]=1359711
pop["Maine"]=1344212
pop["Montana"]=1068778
pop["Rhode Island"]=1059361
pop["Delaware"]=973764
pop["South Dakota"]=884659
pop["North Dakota"]=762062
pop["Alaska"]=731545
pop["Vermont"]=623989
pop["Wyoming"]=578759

# find initial number of infected and recovered individuals
# also calculate average death rate
import csv

inf = {}
rec = {}
death_rate = {}
for state in pop:
    death_rate[state] = 0
    
with open("/kaggle/input/2021spring-cs145-ucla-covid19-prediction/train_trendency.csv", "r") as csv_file:
    reader = csv.reader(csv_file)

    first_row = True
    for row in reader:
        if first_row:
            first_row = False
            continue
        if row[3] != '':
            inf[row[1]] = int(row[3])
        val = row[5].split('.')
        if row[5] != '':
            val = row[5].split('.')
            rec[row[1]] = int(val[0])
            
        # death rate calculation
        state = row[1]
        death_rate[state] = int(row[4]) / int(row[3])
           
# fix data for states that don't have recovered info
avg = 0
for state in rec:
    proportion = rec[state] / inf[state]
    proportion /= len(rec)
    avg += proportion
    
for state in pop:
    if not (state in rec):
        rec[state] = round(inf[state] * avg)

# make calculations
from scipy.integrate import odeint

beta, gamma = 0.05, 1./5
t = np.linspace(0,30,30)

inputs = {}
for state in pop:
    susceptible = pop[state] - inf[state] - rec[state]
    inputs[state] = susceptible, inf[state], rec[state]
    
calc_s = {}
calc_i = {}
calc_r = {}
for state in pop:
    ret = odeint(deriv, inputs[state], t, args=(pop[state], beta, gamma))
    calc_s[state], calc_i[state], calc_r[state] = ret.T

# format output CSV
out_csv = "ID,Confirmed,Deaths\n"
with open("/kaggle/input/2021spring-cs145-ucla-covid19-prediction/test.csv", "r") as csv_file:
    test = csv.reader(csv_file)
    
    first_row = True
    i=0
    for row in test:
        if first_row:
            first_row = False
            continue
            
        state = row[1]
        cumulative_infected = pop[state] - calc_s[state][i]
        row[3] = cumulative_infected
        
        deaths = cumulative_infected * death_rate[state]
        row[4] = deaths
        
        out_csv = out_csv + row[0] + "," + str(cumulative_infected) + "," + str(deaths) + "\n"
        if state == "Wyoming":
            i = i+1

out_file = open("/kaggle/working/submission.csv", "w")
out_file.write(out_csv)
