# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 13:01:50 2022

@author: Prachi Prakash
"""
# Optimal resource allocation

# Importing libraries
import pandas as pd
import pulp as plp
import numpy as np

location_df  = pd.DataFrame({'location': ['location1', 'location2', 'location3'],
                             'max_resource':[500, 600, 250]
                             })
work_df  = pd.DataFrame({'work': ['work1', 'work2'],
                             'min_resource':[550, 300]
                             })
resource_cost = np.array([[150,200], [220,310], [210,440]])

# Minimize cost

# 1. Define model
model = plp.LpProblem("Resource_allocation_prob", plp.LpMinimize)

# Obective function = 150*R00 + 200*R01 + 220*R10 + 310*R11 + 210*R20 + 440R21
# 2. Creating obective function
no_of_location = location_df.shape[0]
no_of_work = work_df.shape[0]

x_vars_list = []
for l in range(1,no_of_location+1):
    for w in range(1,no_of_work+1):
        temp = str(l)+str(w)
        x_vars_list.append(temp)

x_vars = plp.LpVariable.matrix("R", x_vars_list, cat = "Integer", lowBound = 0)
final_allocation = np.array(x_vars).reshape(3,2)
print(final_allocation)

objective_function = plp.lpSum(final_allocation*resource_cost)
model +=  objective_function

# 3. Adding constraints
# There are two constraints - 
# R11 + R12 <= 500
# R21 + R22 <= 600
# R31 + R32 <= 250

# R11 + R21 + R31 >= 550
# R12 + R22  R32 >= 300

for l1 in range(no_of_location):
    model += plp.lpSum(final_allocation[l1][w1] for w1 in range(no_of_work)) <= location_df['max_resource'].tolist()[l1]
        
for w2 in range(no_of_work):
    model += plp.lpSum(final_allocation[l2][w2] for l2 in range(no_of_location)) >= work_df['min_resource'].tolist()[w2]
    
print(model)    

# 4. Run the model
model.solve()

# 5. Checking the status
status = plp.LpStatus[model.status]
print(status)

# Checking optimal resource allocation
print("Optimal overall resouce cost: ",str(plp.value(model.objective)))

for each in model.variables():
    print("Optimal cost of ", each, ": "+str(each.value()))