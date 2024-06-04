import numpy as np
import pandas as pd
import math
from gurobipy import Model, GRB, Env, quicksum
import matplotlib.pyplot as plt
import requests
import json
import threading
import ast


# Discord notification function
def send_discord_notification(message):
    url = "https://discord.com/api/webhooks/1245288786024206398/ZQEM6oSRWOYw0DV9_3WUNGYIk7yZQ-M1OdsZU6J3DhUKhZ-qmi8ecqJRAVBRqwpJt0q8"
    data = {"content": message}
    response = requests.post(
        url, data=json.dumps(data), headers={"Content-Type": "application/json"}
    )

    if response.status_code == 204:
        print("Notification sent successfully.")
    else:
        print("Failed to send notification.")


# Stage 1: Loading & Preprocessing Data
def load_and_preprocess_data(new_employee_path, new_task_path):
    try:
        global employees
        global tasks
        global employee_skills_df
        global task_df
        global task_skills_df
        global story_points
        global company_tasks

        employee_skills_df = pd.read_csv(new_employee_path, index_col='employee_id').fillna(0)
        employee_skills_df.drop(columns=['no', 'Role'], inplace=True, errors='ignore')

        employees = employee_skills_df.index.tolist()
        skills_name = employee_skills_df.columns[1:].tolist()

        task_df = pd.read_csv(new_task_path, index_col='task_id').fillna(0)

        tasks = task_df.index.tolist()
        company_names = list(set(task_df['project_id']))
        
        story_points = task_df['story_points'].to_dict()

        # convert to dictionary each company and its task
        company_tasks = {}
        for company in company_names:
            company_tasks[company] = task_df[task_df['project_id'] == company].index.tolist()

        # sort the company tasks from C1 to C5
        company_tasks = dict(sorted(company_tasks.items()))

        task_skills_df = task_df.drop(columns=['project_id', 'story_points'])
        task_skills_df.head()

        print("Stage 1: Data Loading & Preprocessing Successful")

    except Exception as e:
        print(f"Stage 1: Data Loading & Preprocessing Failed with error: {str(e)}")
        raise

# Stage 2: Calculating Similarity Error
def calculate_similarity_error():
    try:
        def custom_we(v1, v2, a=0.5):
            diff = v1 - v2
            w = 1 / (1 + a * np.maximum(0, diff))
            return w

        def euclidean_similarity(emp, task):
            sum = 0
            for index, metric in enumerate(emp):
                if task[index] > 0:
                    w = custom_we(emp[index], task[index])
                    sum += w * ((emp[index] - task[index])**2)
                else:
                    sum += 0
            return math.sqrt(sum)

        global euclidean_similarity_score
        euclidean_similarity_score = {}
        count_no_match = 0

        for i in tasks:
            task_skills = task_skills_df.loc[i]
            for j in employees:
                employee_skills = employee_skills_df.loc[j]

                 # Filter skills to consider only those present in both project requirements and employee skills
                common_skills = [skill for skill in employee_skills.index if skill in task_skills.index]
                
                # check if there's at least one skill matching
                if common_skills:
                    # calculate weighted euclidean distance for common skills
                    euclidean_similarity_score[(i, j)] = euclidean_similarity(employee_skills[common_skills], task_skills[common_skills])
                    euclidean_similarity_score[(i, j)] = 1 / (1 + euclidean_similarity_score[(i, j)])
                else:
                    count_no_match += 1

        # euclidean_similarity_score_df = pd.DataFrame.from_dict(euclidean_similarity_score, orient='index')

        print("Stage 2: Calculating Similarity Error Successful")
        return euclidean_similarity_score
    except Exception as e:
        print(f"Stage 2: Calculating Similarity Error Failed with error: {str(e)}")
        raise

# Stage 3: Construct Model, Build Decision Variable, Set Constraints
def construct_model_build_variables_set_constraints():
    try:
        options = {
            "WLSACCESSID": "03cdd75c-8718-4560-a512-4a72cd7912ec",
            "WLSSECRET": "36cd9392-3529-4b70-bdc1-d674430af621",
            "LICENSEID": 2521689,
        }
        
        global model
        global max_employee_workload
        global x
        global y
        global max_workload

        env = Env(params=options)
        model = Model("task_assignment", env=env)
        
        max_employee_workload = 20

        # Set Gurobi parameters to improve performance
        model.setParam("Presolve", 2)  # Aggressive presolve
        model.setParam("MIPFocus", 1)  # Focus on improving the best bound
        model.setParam("MIPGap", 0.01)  # 1% optimality gap
        model.setParam("Heuristics", 0.1)  # Increase heuristics effort
        model.setParam("Threads", 2)  # Use 8 threads, adjust based on your CPU
        
        # Create decision variables for x
        x = {}
        for k, task in company_tasks.items():
            for i in task:
                for j in employees:
                    x[(i, j, k)] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}_{k}')
        
        # Decision variable y represents cardinality of each employee and company
        y = {}
        for j in employees:
            for k in company_tasks.keys():
                y[(j, k)] = model.addVar(vtype=GRB.BINARY, name=f'y_{j}_{k}')
        
        # Decision variable max_workload
        max_workload = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_employee_workload, name='max_workload')
        
        # Constraint 1: Each task assigned to one talent
        for k, task in company_tasks.items():
            for i in task:
                model.addConstr(quicksum(x[(i, j, k)] for j in employees) == 1, name=f'task_{i}_assigned_once')
        
        # Pre-processing constraint 2: If task i is assigned to employee j, then y[j, k] = 1
        for j in employees:
            for k, task in company_tasks.items():
                # Create a temporary list to hold the sum of x[i][j][k] for all i
                temp_sum = []
                for i in task:
                    temp_sum.append(x[(i, j, k)])

                # Add a constraint to the model: y[j][k] is 1 if the sum of x[i][j][k] for all i is > 0, and 0 otherwise
                model.addConstr(quicksum(temp_sum) >= 1 - (1 - y[(j, k)]), name=f'y_{j}_{k}_enforced_if_any_task_assigned')
                model.addConstr(quicksum(temp_sum) <= (len(task) * y[(j, k)]), name=f'y_{j}_{k}_0_if_no_task_assigned')
        
        # Create constraint 2: each employee can only work on one task
        for j in employees:
            # The sum of y[j][k] for all companies (k) should be <= 1
            model.addConstr(quicksum(y[(j, k)] for k in company_tasks.keys()) <= 1, name=f'employee_{j}_single_task')
        
        # Constraint 3: Employee workload doesn't exceed the capacity
        for j in employees:
            model.addConstr(quicksum(story_points[i] * x[(i, j, k)] for k, tasks in company_tasks.items() for i in tasks) <= max_employee_workload, name=f'employee_{j}_workload_capacity')
        
        # Constraint 4: max_workload is greater than or equal to the workload of each employee
        for j in employees:
            model.addConstr(max_workload >= quicksum(story_points[i] * x[(i, j, k)] for k, tasks in company_tasks.items() for i in tasks), name=f'max_workload_{j}')
        
        print("Weighted Euclidean - Stage 3: Construct Model, Build Variables, Set Constraints Successful")
        # return model, x, y, max_workload, max_employee_workload
    except Exception as e:
        print(f"Weighted Euclidean - Stage 3: Construct Model, Build Variables, Set Constraints Failed with error: {str(e)}")
        raise

# Stage 4: Objective 1 Optimization
def objective1_optimization():
    try:
        I = []

        global x_hat_obj1
        for j in employees:
            obj1 = 1 - quicksum(y[j, k] for k in company_tasks.keys())
            I.append(obj1)

        global I_total_idle_employee
        I_total_idle_employee = quicksum(I)
        model.setObjective(I_total_idle_employee, GRB.MINIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            # x_hat_obj1 = get_results(model, employees, company_tasks, "Objective 1")
            # save_x_hat(x_hat_obj1, 'optimization1_results.xlsx')
            mu_Z_star[1] = model.ObjVal
            x_hat_obj1 = {}

            for j in employees:
                task = []
                sim = []
                sp = 0
                wasted_sp = 0
                comp = []

                for k, tasks in company_tasks.items():
                    for i in tasks:
                        if x[i, j, k].X == 1:
                            print(f'Task {i} assigned to Employee {j}')
                            print(f'Company\t\t\t: {k}')
                            print(f'Story Points\t\t: {story_points[i]}')
                            print(f"Similarity score\t: {euclidean_similarity_score[i, j]:.10f}\n")

                            task.append(i)
                            sim.append(euclidean_similarity_score[i, j])
                            comp.append(k)
                            sp += story_points[i]

                if sp > 0:
                    wasted_sp = max_employee_workload - sp
                    x_hat_obj1[j] = comp, task, sp, wasted_sp, sim

            print("Stage 4: Objective 1 Optimization Successful")
        else:
            print("Stage 4: Objective 1 Optimization No Solution Found!")
            x_hat_obj1 = {}

        # Convert dictionary to DataFrame
        global result_obj1
        result_obj1 = pd.DataFrame([(key, value[0], value[1], value[2], value[3], value[4]) for key, value in x_hat_obj1.items()],
                            columns=['employee', 'company', 'assigned_task', 'sum_sp', 'wasted_sp', 'similarity_score'])


        # Set 'company' as index
        result_obj1.set_index('employee', inplace=True)
        result_obj1.to_csv("./output/result_obj1.csv")


    except Exception as e:
        print(f"Stage 4: Objective 1 Optimization Failed with error: {str(e)}")
        raise


# Stage 5: Objective 2 Optimization
def objective2_optimization():
    try:
        global E_total_similarity_score
        global x_hat_obj2
        E_total_similarity_score = 0
        for k, tasks in company_tasks.items():
            E_total_similarity_score += quicksum(euclidean_similarity_score[i, j] * x[i, j, k] for i in tasks for j in employees)

        model.setObjective(E_total_similarity_score, GRB.MINIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            mu_Z_star[2] = model.ObjVal
            x_hat_obj2 = {}

            for j in employees:
                task = []
                sim = []
                sp = 0
                wasted_sp = 0
                comp = []

                for k, tasks in company_tasks.items():
                    for i in tasks:
                        if x[i, j, k].X == 1:
                            print(f'Task {i} assigned to Employee {j}')
                            print(f'Company\t\t\t: {k}')
                            print(f'Story Points\t\t: {story_points[i]}')
                            print(f"Similarity score\t: {euclidean_similarity_score[i, j]:.10f}\n")

                            task.append(i)
                            sim.append(euclidean_similarity_score[i, j])
                            comp.append(k)
                            sp += story_points[i]

                if sp > 0:
                    wasted_sp = max_employee_workload - sp
                    x_hat_obj2[j] = comp, task, sp, wasted_sp, sim

            print("Stage 5: Objective 2 Optimization Successful")
        else:
            print("Stage 5: Objective 2 Optimization No Solution Found!")
            x_hat_obj2 = {}

        # Convert dictionary to DataFrame
        global result_obj2
        result_obj2 = pd.DataFrame([(key, value[0], value[1], value[2], value[3], value[4]) for key, value in x_hat_obj2.items()],
                            columns=['employee', 'company', 'assigned_task', 'sum_sp', 'wasted_sp', 'similarity_score'])

        # Set 'company' as index
        result_obj2.set_index('employee', inplace=True)
        result_obj2.to_csv("./output/result_obj2.csv")

    except Exception as e:
        print(f"Stage 5: Objective 2 Optimization Failed with error: {str(e)}")
        raise

# Stage 6: Objective 3 Optimization
def objective3_optimization():
    try:
        global x_hat_obj3
        model.setObjective(max_workload, GRB.MINIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            mu_Z_star[3] = model.ObjVal
            x_hat_obj3 = {}

            for j in employees:
                task = []
                sim = []
                sp = 0
                wasted_sp = 0
                comp = []

                for k, tasks in company_tasks.items():
                    for i in tasks:
                        if x[i, j, k].X == 1:
                            print(f'Task {i} assigned to Employee {j}')
                            print(f'Company\t\t\t: {k}')
                            print(f'Story Points\t\t: {story_points[i]}')
                            print(f"Similarity score\t: {euclidean_similarity_score[i, j]:.10f}\n")

                            task.append(i)
                            sim.append(euclidean_similarity_score[i, j])
                            comp.append(k)
                            sp += story_points[i]

                if sp > 0:
                    wasted_sp = max_employee_workload - sp
                    x_hat_obj3[j] = comp, task, sp, wasted_sp, sim

            print("Stage 6: Objective 3 Optimization Successful")
        else:
            print("Stage 6: Objective 3 Optimization No Solution Found!")
            x_hat_obj3 = {}

        # Convert dictionary to DataFrame
        global result_obj3
        result_obj3 = pd.DataFrame([(key, value[0], value[1], value[2], value[3], value[4]) for key, value in x_hat_obj3.items()],
                            columns=['employee', 'company', 'assigned_task', 'sum_sp', 'wasted_sp', 'similarity_score'])

        # Set 'company' as index
        result_obj3.set_index('employee', inplace=True)
        result_obj3.to_csv("./output/result_obj3.csv")


    except Exception as e:
        print(f"Stage 6: Objective 3 Optimization Failed with error: {str(e)}")
        raise

# Stage 7: MOO Removing Objective 1 Optimization
def moo_rm_obj1_optimization():
    try:
        global x_hat_rm_obj1
        beta = 0.2
        theta = 0.8
        model.setObjective((beta * E_total_similarity_score) + (theta * max_workload), GRB.MINIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            mu_Z_star[4] = model.ObjVal
            x_hat_rm_obj1 = {}

            for j in employees:
                task = []
                sim = []
                sp = 0
                wasted_sp = 0
                comp = []

                for k, tasks in company_tasks.items():
                    for i in tasks:
                        if x[i, j, k].X == 1:
                            print(f'Task {i} assigned to Employee {j}')
                            print(f'Company\t\t\t: {k}')
                            print(f'Story Points\t\t: {story_points[i]}')
                            print(f"Similarity score\t: {euclidean_similarity_score[i, j]:.10f}\n")

                            task.append(i)
                            sim.append(euclidean_similarity_score[i, j])
                            comp.append(k)
                            sp += story_points[i]

                if sp > 0:
                    wasted_sp = max_employee_workload - sp
                    x_hat_rm_obj1[j] = comp, task, sp, wasted_sp, sim

            print("Stage 7: MOO removing obj 1 Optimization Successful")
        else:
            print("Stage 7: MOO removing obj 1 Optimization No Solution Found!")
            x_hat_rm_obj1 = {}

        # Convert dictionary to DataFrame
        global result_rm_obj1
        result_rm_obj1 = pd.DataFrame([(key, value[0], value[1], value[2], value[3], value[4]) for key, value in x_hat_rm_obj1.items()],
                            columns=['employee', 'company', 'assigned_task', 'sum_sp', 'wasted_sp', 'similarity_score'])

        # Set 'company' as index
        result_rm_obj1.set_index('employee', inplace=True)
        result_rm_obj1.to_csv("./output/result_rm_obj1.csv")


    except Exception as e:
        print(f"Stage 7: MOO Removing Objective 1 Optimization Failed with error: {str(e)}")
        raise

# Stage 8: MOO Removing Objective 2 Optimization
def moo_rm_obj2_optimization():
    try:
        global x_hat_rm_obj2
        alpha = 0.2
        theta = 0.8
        model.setObjective((alpha * I_total_idle_employee) + (theta * max_workload), GRB.MINIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            mu_Z_star[5] = model.ObjVal
            x_hat_rm_obj2 = {}

            for j in employees:
                task = []
                sim = []
                sp = 0
                wasted_sp = 0
                comp = []

                for k, tasks in company_tasks.items():
                    for i in tasks:
                        if x[i, j, k].X == 1:
                            print(f'Task {i} assigned to Employee {j}')
                            print(f'Company\t\t\t: {k}')
                            print(f'Story Points\t\t: {story_points[i]}')
                            print(f"Similarity score\t: {euclidean_similarity_score[i, j]:.10f}\n")

                            task.append(i)
                            sim.append(euclidean_similarity_score[i, j])
                            comp.append(k)
                            sp += story_points[i]

                if sp > 0:
                    wasted_sp = max_employee_workload - sp
                    x_hat_rm_obj2[j] = comp, task, sp, wasted_sp, sim

            print("Stage 8: MOO Removing Objective 2 Optimization Successful")
        else:
            print("Stage 8: MOO Removing Objective 2 Optimization No Solution Found!")
            x_hat_rm_obj2 = {}

        # Convert dictionary to DataFrame
        global result_rm_obj2
        result_rm_obj2 = pd.DataFrame([(key, value[0], value[1], value[2], value[3], value[4]) for key, value in x_hat_rm_obj2.items()],
                            columns=['employee', 'company', 'assigned_task', 'sum_sp', 'wasted_sp', 'similarity_score'])

        # Set 'company' as index
        result_rm_obj2.set_index('employee', inplace=True)
        result_rm_obj2.to_csv("./output/result_rm_obj2.csv")

    except Exception as e:
        print(f"Stage 8: MOO Removing Objective 2 Optimization Failed with error: {str(e)}")
        raise

# Stage 9: MOO Removing Objective 3 Optimization
def moo_rm_obj3_optimization():
    try:
        global x_hat_rm_obj3
        alpha = 0.1
        beta = 0.9
        
        model.setObjective((alpha * I_total_idle_employee) + (beta * E_total_similarity_score), GRB.MINIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            mu_Z_star[6] = model.ObjVal
            x_hat_rm_obj3 = {}

            for j in employees:
                task = []
                sim = []
                sp = 0
                wasted_sp = 0
                comp = []

                for k, tasks in company_tasks.items():
                    for i in tasks:
                        if x[i, j, k].X == 1:
                            print(f'Task {i} assigned to Employee {j}')
                            print(f'Company\t\t\t: {k}')
                            print(f'Story Points\t\t: {story_points[i]}')
                            print(f"Similarity score\t: {euclidean_similarity_score[i, j]:.10f}\n")

                            task.append(i)
                            sim.append(euclidean_similarity_score[i, j])
                            comp.append(k)
                            sp += story_points[i]

                if sp > 0:
                    wasted_sp = max_employee_workload - sp
                    x_hat_rm_obj3[j] = comp, task, sp, wasted_sp, sim

            print("Stage 8: MOO Removing Objective 2 Optimization Successful")
        else:
            print("Stage 8: MOO Removing Objective 2 Optimization No Solution Found!")
            x_hat_rm_obj3 = {}

        # Convert dictionary to DataFrame
        global result_rm_obj3
        result_rm_obj3 = pd.DataFrame([(key, value[0], value[1], value[2], value[3], value[4]) for key, value in x_hat_rm_obj3.items()],
                            columns=['employee', 'company', 'assigned_task', 'sum_sp', 'wasted_sp', 'similarity_score'])

        # Set 'company' as index
        result_rm_obj3.set_index('employee', inplace=True)
        result_rm_obj3.to_csv("./output/result_rm_obj3.csv")

        
    except Exception as e:
        print(f"Stage 9: MOO Removing Objective 3 Optimization Failed with error: {str(e)}")
        raise

# Stage 9: MOO Removing Objective 3 Optimization
def moo_optimization():
    try:
        global x_hat_moo

        # Define weight dictionary for each objective
        Weight = {
            1: 0.05,
            2: 0.85,
            3: 0.1,
        }

        # Define the deviation plus and minus variables
        d_plus = {}
        d_minus = {}

        # Add variables for d_plus and d_minus with specific conditions
        for i in range(1, 4):
            if i != 3:
                d_plus[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"d_plus_{i}")
            if i != 1:
                d_minus[i] = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, name=f"d_minus_{i}"
                )

        # Set specific variables to zero
        d_minus[1] = 0
        d_plus[3] = 0

        mu_Z = {1: I_total_idle_employee, 2: E_total_similarity_score, 3: max_workload}

        mu_Z_star_obj = mu_Z_star.copy()
        for i, value in mu_Z_star_obj.items():
            mu_Z_star_obj[i] = 1 / value if value != 0 else 0

        # This constraint is bound for each objective connected with each other
        for k, w in Weight.items():
            if w != 0:
                model.addConstr(mu_Z[k] - d_plus[k] + d_minus[k] == mu_Z_star[k])

        # Define D = sum k=1 to 3 ((W_plus_k * d_plus_k) + (W_minus_k * d_minus_k)) / mu_Z_star iterate
        D = quicksum(
            (
                (Weight[i] * (d_plus[i] + d_minus[i])) * mu_Z_star_obj[i]
                for i in range(1, 4)
            )
        )


        model.setObjective(D, GRB.MINIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            mu_Z_star[4] = model.ObjVal
            x_hat_moo = {}

            for j in employees:
                task = []
                sim = []
                sp = 0
                wasted_sp = 0
                comp = []

                for k, tasks in company_tasks.items():
                    for i in tasks:
                        if x[i, j, k].X == 1:
                            print(f'Task {i} assigned to Employee {j}')
                            print(f'Company\t\t\t: {k}')
                            print(f'Story Points\t\t: {story_points[i]}')
                            print(f"Similarity score\t: {euclidean_similarity_score[i, j]:.10f}\n")

                            task.append(i)
                            sim.append(euclidean_similarity_score[i, j])
                            comp.append(k)
                            sp += story_points[i]

                if sp > 0:
                    wasted_sp = max_employee_workload - sp
                    x_hat_moo[j] = comp, task, sp, wasted_sp, sim

            print("Stage 7: MOO Removing Objective 2 Optimization Successful")
        else:
            print("Stage 7: MOO Removing Objective 2 Optimization No Solution Found!")
            x_hat_moo = {}

        # Convert dictionary to DataFrame
        global result_moo
        result_moo = pd.DataFrame([(key, value[0], value[1], value[2], value[3], value[4]) for key, value in x_hat_moo.items()],
                            columns=['employee', 'company', 'assigned_task', 'sum_sp', 'wasted_sp', 'similarity_score'])

        # Set 'company' as index
        result_moo.set_index('employee', inplace=True)
        result_moo.to_csv("./output/result_moo.csv")

        
    except Exception as e:
        print(f"Stage 7: MOO Removing Objective 3 Optimization Failed with error: {str(e)}")
        raise



# Import library
import json
import requests
import datetime
import threading
import numpy as np
import pandas as pd
import gurobipy as gp
import matplotlib.pyplot as plt
from gurobipy import GRB, quicksum
from competency_assessment import CompetencyAssessment

full = False

if full:
    # Data Fix
    EMPLOYEE_PATH = "./data/fixed_data_employee.csv"
    TASK_PATH = "./data/fixed_data_task.csv"
else:
    # # Mini Data
    EMPLOYEE_PATH = "/content/drive/MyDrive/Tadika Mesra Bunga Matahari/#1 Optimization Problem/project1_task-assignment/data/experimental/mini_data - employee.csv"
    TASK_PATH = "/content/drive/MyDrive/Tadika Mesra Bunga Matahari/#1 Optimization Problem/project1_task-assignment/data/experimental/mini_data - task.csv"
    # EMPLOYEE_PATH = "/content/employee_minmax.csv"
    # TASK_PATH = "/content/task_minmax.csv"

    # EMPLOYEE_PATH = '/content/drive/MyDrive/Tadika Mesra Bunga Matahari/#1 Optimization Problem/project1_task-assignment/data/experimental/medium_data_employee.csv'
    # TASK_PATH = '/content/drive/MyDrive/Tadika Mesra Bunga Matahari/#1 Optimization Problem/project1_task-assignment/data/experimental/medium_data_task.csv'
    # EMPLOYEE_PATH = "/content/employee_minmax.csv"
    # TASK_PATH = "/content/task_minmax.csv"




def s1_data_structure_CA():
    """# 1. Define the Data Structure"""

    try:
        # Run this if the data in Local/Repository
        new_employee_path = EMPLOYEE_PATH
        new_task_path = TASK_PATH

        """## 1.1. Pre-Processing: Employee Data"""

        # Read data
        employee_skills_df = pd.read_csv(new_employee_path, index_col="employee_id")
        employee_skills_df.drop(columns=["No", "Role"], inplace=True, errors="ignore")

        employees = employee_skills_df.index.tolist()
        skills_name = employee_skills_df.columns[1:].tolist()

        """## 1.2. Pre-Processing: Task Data"""

        task_df = pd.read_csv(new_task_path, index_col="task_id")

        tasks = task_df.index.tolist()
        company_names = list(set(task_df["project_id"]))
        story_points = task_df["story_points"].to_dict()

        """## 1.3. Group the task data by company/project"""

        # convert to dictionary each company and its task
        company_tasks = {}

        for company in company_names:
            company_tasks[company] = task_df[
                task_df["project_id"] == company
            ].index.tolist()

        # sort the company tasks from C1 to C5
        company_tasks = dict(sorted(company_tasks.items()))

        company_tasks_df = pd.DataFrame.from_dict(company_tasks, orient="index")

        """## 1.4. Pre-Processing: Competency Assessment

        First, create RCD-ACD Dataframe that we get from Task Dataframe for RCD and from Employee Dataframe for ACD.

        ### 1.4.1 Required Competence Data
        """

        rcd_df = task_df.drop(columns=["project_id", "story_points"])
        rcd_df = rcd_df.fillna(0)

        """### 1.4.2 Acquired Competence Data"""

        # create a copy of the original DataFrame
        acd_df = employee_skills_df.copy()
        acd_df = acd_df.fillna(0)

        """### 1.4.3 Fit the Data"""

        ca = CompetencyAssessment(rcd_df, acd_df)
        qs, info = ca.fit()

        """### 1.4.4 Qualification Space"""

        """### 1.4.5 Sorted MSG Score for All Tasks"""

        score = ca.rank_MSG(qs)
        score_df = pd.DataFrame.from_dict(score, orient="index")

        # Export the score dictionary to CSV
        score_df.to_csv("./output_CA/score.csv")

        return employees, skills_name, tasks, story_points, company_tasks, score

    except Exception as e:
        print(f"An error occured in s1_data_structure_CA: {e}")
        print(f"An error occurred in s1_data_structure_CA: {e}")
        return [], [], [], {}, {}, {}


"""#### Generic Function"""


# Extracting and printing the results
def get_employee_tasks(
    j, company_tasks, model, score, story_points, max_employee_workload
):
    task = []
    sim = []
    comp = []
    sp = 0

    for k, tasks in company_tasks.items():
        for i in tasks:
            if model.getVarByName(f"x_{i}_{j}_{k}").X == 1:
                print(f"Task {i} assigned to Employee {j}")
                print(f"Company\t\t\t: {k}")
                print(f"Story Points\t\t: {story_points[i]}")
                print(f"Metrics score\t: {score[j][i]:.10f}\n")

                task.append(i)
                sim.append(score[j][i])
                comp.append(k)
                sp += story_points[i]

    wasted_sp = max_employee_workload - sp if sp > 0 else 0
    return comp, task, sp, wasted_sp, sim


def s2_construct_model():
    """# 2. Construct the Model"""

    try:

        # Create an environment with WLS license
        params = {
            "WLSACCESSID": "03cdd75c-8718-4560-a512-4a72cd7912ec",
            "WLSSECRET": "36cd9392-3529-4b70-bdc1-d674430af621",
            "LICENSEID": 2521689,
        }
        env = gp.Env(params=params)

        # Create the model within the Gurobi environment
        model = gp.Model(name="task_assignment", env=env)

        # Set Gurobi parameters to improve performance
        model.setParam("Presolve", 2)  # Aggressive presolve
        model.setParam("MIPFocus", 1)  # Focus on improving the best bound
        model.setParam("MIPGap", 0.01)  # 1% optimality gap
        model.setParam("Heuristics", 0.8)  # Increase heuristics effort
        model.setParam("Threads", 2)  # Use 8 threads, adjust based on your CPU

        return model

    except Exception as e:
        print(f"An error occured in s2_construct_model: {e}")
        print(f"An error occurred in s2_construct_model: {e}")
        return None


def s3_decision_variable(model, tasks, employees, company_tasks):
    """# 3. Build the Decision Variable

    We have 3 sets:

    $$
    \text{sets} = \begin{cases}
    I &: \text{set of tasks} \\
    J &: \text{set of employees} \\
    K &: \text{set of projects}
    \\end{cases}
    $$

    Next, we define parameters, scalars, and data structures. Let:

    $$
    \begin{align*}
    i & = \text{task } i \\
    j & = \text{employee } j \\
    k & = \text{project } k \\
    s_i & = \text{story points of task } i \\
    e_{ij} & = \text{similarity skills of employee } j \text{ for task } i \\
    \\end{align*}
    $$

    **Decision Variables:**

    $$
    \begin{align*}
    x_{ijk} & = \text{Binary variable indicating whether employee } j \text{ is assigned to task } k \text{ for day } i \\
    y_{jk} & = \text{Binary variable indicating whether employee } j \text{ is assigned to any task from company } k \\
    \\end{align*}
    $$

    """

    try:
        if full:
            max_employee_workload = 20
        else:
            max_employee_workload = 20

        # Create decision variables for x and y
        x = {}
        for k, task in company_tasks.items():
            for i in task:
                for j in employees:
                    x[(i, j, k)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

        # Decision variable y to represent cardinality of each employee and company
        y = {}
        for j in employees:
            for k in company_tasks.keys():
                y[(j, k)] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{k}")

        z = {}
        for i in tasks:
            for j in employees:
                z[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")

        # Decision variable for max workload
        max_workload = model.addVar(
            vtype=GRB.INTEGER, lb=0, ub=max_employee_workload, name="max_workload"
        )

        # Integrate new variables
        model.update()

        return x, y, z, max_employee_workload, max_workload

    except Exception as e:
        print(f"An error occured in s3_decision_variable: {e}")
        print(f"An error occurred in s3_decision_variable: {e}")
        return {}, {}, {}, 0, None


def s4_constraint(model, x, y, z, employees, company_tasks, story_points, max_workload):
    """# 4. Subject to the Constraint

    ## 4.1. Constraint 1: Each Task is Assigned to One Employee

    $$
    \\sum _{j\\in J}\\:x_{ijk}\\:=\\:1 \\quad \\forall i \\in k, \\: k \\in K
    $$

    """

    try:
        # constraint 1: each task assigned to one talent
        for k, task in company_tasks.items():
            for i in task:
                model.addConstr(quicksum(x[(i, j, k)] for j in employees) == 1)

        """## 4.2. Constraint 2: Each employee works for one company at a time

        Pre-Processing for Constraint 2:

        $$
        \\sum _{i\\in I_k}x_{ijk} > 0 \\: \\rightarrow \\: y_{jk}=1 \\quad \\forall j\\in J, \\: k\\in K\\:
        $$

        """

        # pre-processing constraint 2
        for j in employees:
            for k, task in company_tasks.items():
                # Use quicksum to sum up x[i][j][k] for all i
                temp_sum = quicksum(x[i, j, k] for i in task)

                # Add a constraint to the model: y[j][k] is 1 if the sum of x[i][j][k] for all i is > 0, and 0 otherwise
                model.addGenConstrIndicator(
                    y[j, k], True, temp_sum, GRB.GREATER_EQUAL, 1
                )
                model.addGenConstrIndicator(y[j, k], False, temp_sum, GRB.LESS_EQUAL, 0)

        """$$
        \\sum _{k\\in K}y_{jk}\\le 1 \\quad \\forall j\\in J
        $$

        """

        # create constraint 2: each employee can only work on one task
        for j in employees:
            # The sum of y[j][k] for all companies (k) should be <= 1
            model.addConstr(quicksum(y[(j, k)] for k in company_tasks.keys()) <= 1)

        """## 4.3. Constraint 3: Employee workload doesn't exceed the capacity

        $$
        \\sum _{i \\in I} s_i \\cdot x_{ijk} \\le max\\_workload \\quad \\forall j\\in J, \\: k \\in K
        $$

        """

        for j in employees:
            for k, tasks in company_tasks.items():
                model.addConstr(
                    quicksum(story_points[i] * x[(i, j, k)] for i in tasks)
                    <= max_workload
                )

        """## 4.4 Constraint 4: Maximum workload is greater than or equal to the workload of each employee For Objective 3

        $$
        max\\_workload \\ge \\sum_{i \\in I} \\sum_{k \\in K} s_i\\cdot x_{ijk}, \\quad \\forall j\\in J\\:\\:
        $$

        """

        # constraint 4: max_workload is greater than or equal to the workload of each employee
        for j in employees:
            model.addConstr(
                max_workload
                >= quicksum(
                    story_points[i] * x[i, j, k]
                    for k, tasks in company_tasks.items()
                    for i in tasks
                )
            )

        # # create constraint 5: each task can only assigned to one employee
        for i in tasks:
            model.addConstr(quicksum(z[(i, j)] for j in employees) <= 1)

        for k, tasks in company_tasks.items():
            for i in tasks:
                for j in employees:
                    model.addGenConstrIndicator(x[i, j, k], True, z[i, j], GRB.EQUAL, 1)
                    model.addGenConstrIndicator(z[i, j], True, y[j, k], GRB.EQUAL, 1)

    except Exception as e:
        print(f"An error occured in s4_constraint: {e}")
        print(f"An error occurred in s4_constraint: {e}")


def s5_objective1(
    model,
    employees,
    company_tasks,
    y,
    score,
    story_points,
    max_employee_workload,
    mu_Z_star,
):
    """# 5. Single Objective Approach: 1) Minimize The Idle Employee
    ## 5.1. Set The Objective Model

    $$
    \\mu _{Z_1} = min.\\:I_j=\\sum _{j\\in \\:J}\\:\\left(1\\:-\\:\\sum _{k\\in \\:K}\\:y_{jk}\\right) \\quad \\tag{1}
    $$
    """

    try:
        # objective 1
        idle = []

        for j in employees:
            idle.append(1 - quicksum(y[j, k] for k in company_tasks.keys()))

        mu_Z_1 = quicksum(idle)

        # single objective 1
        model.setObjective(mu_Z_1, GRB.MINIMIZE)

        """## 5.2. Solve The Model of Objective $(1)$"""

        # solve the model
        model.optimize()

        """### 5.2.1 Print The Solver Results"""

        # Check and process the solution
        if model.status == GRB.OPTIMAL:
            print("Solution Found!")
            print(f"Obj. Value 1 i.e. Total Idle Employees: {model.ObjVal}\n")
            mu_Z_star[1] = model.ObjVal

            x_hat_1 = {}
            for j in employees:
                result = get_employee_tasks(
                    j, company_tasks, model, score, story_points, max_employee_workload
                )
                if len(result[1]) > 0:
                    x_hat_1[j] = result
        else:
            print("No Solution Found!")
            x_hat_1 = {}

        """## 5.3. Show the Solver's Result"""

        # Set display options
        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)

        # Convert dictionary to DataFrame and set 'employee' as index
        global result_1
        result_1 = pd.DataFrame.from_dict(
            x_hat_1,
            orient="index",
            columns=[
                "company",
                "assigned_task",
                "sum_sp",
                "wasted_sp",
                "assessment_score",
            ],
        )
        result_1.index.name = "employee"
        result_1.to_csv("./output_CA/result_1.csv")

        """### 5.3.1 Statistics of The Objective"""

        total_employee = len(employees)
        total_sp = sum(story_points.values())
        total_active_employee = len(set(employee for employee in x_hat_1.keys()))
        total_active_sp = sum(value[2] for value in x_hat_1.values())
        total_idle_employee = total_employee - total_active_employee
        total_wasted_sp = total_sp - total_active_sp

        print(f"Total Employee\t\t\t: {total_employee}")
        print(
            f"Total Active Employee\t\t: {total_active_employee}\t{(total_active_employee/total_employee)*100:.2f}%"
        )
        print(
            f"Total Idle Employee\t\t: {total_idle_employee}\t{(total_idle_employee/total_employee)*100:.2f}%\n"
        )
        print(f"Total Story Points\t\t: {total_sp}")
        print(
            f"Total Active Story Points\t: {total_active_sp}\t{(total_active_sp/total_sp)*100:.2f}%"
        )
        print(
            f"Total Wasted Story Points\t: {total_wasted_sp}\t{(total_wasted_sp/total_sp)*100:.2f}%\n"
        )

        """### 5.3.2. Distribution With Respect to the Assessment Score"""

        # timer for auto close plot
        timer = threading.Timer(3, close_plot)
        timer.start()

        # make boxplot for objective 1 with respect to the assessment score
        assessment_score_1 = (
            result_1["assessment_score"].explode().reset_index(drop=True)
        )

        if len(assessment_score_1) != 0:
            assessment_score_1.plot(kind="box")
            plt.title("Assessment Score Boxplot of Objective 1")
            plt.show()
        else:
            print("No data to show")

        return mu_Z_1, mu_Z_star, assessment_score_1

    except Exception as e:
        print(f"An error occured in s5_objective1: {e}")
        print(f"An error occurred in s5_objective1: {e}")
        return None, None, None


def s6_objective2(
    model,
    employees,
    company_tasks,
    z,
    score,
    story_points,
    max_employee_workload,
    mu_Z_star,
):
    """# 6. Single Objective Approach: 2) Maximize The Assessment Score
    ## 6.1. Set The Objective Model

    $$
    \\mu _{Z_2} = max.\\:A_{ij} = \\sum _{i\\in \\:I} \\sum _{j\\in \\:J} \\sum _{k\\in \\:K} \\: e_{ij} \\cdot x_{ijk} \\quad \\tag{2}
    $$
    """

    try:
        # objective 2
        # mu_Z_2 = quicksum(
        #     score[j][i] * x[i, j, k]
        #     for k, tasks in company_tasks.items()
        #     for i in tasks
        #     for j in employees
        # )

        # proposed objective 2
        mu_Z_2 = quicksum(
            score[j][i] * z[i, j]
            for k, tasks in company_tasks.items()
            for i in tasks
            for j in employees
        )

        # single objective 2
        model.setObjective(mu_Z_2, GRB.MAXIMIZE)

        """## 6.2. Solve The Model of Objective $(2)$"""

        # solve the model
        model.optimize()

        """### 6.2.1 Print The Solver Results"""

        # Check and process the solution
        if model.status == GRB.OPTIMAL:
            print("Solution Found!")
            print(f"Obj. Value 2 i.e. Total Score: {model.ObjVal}\n")
            mu_Z_star[2] = model.ObjVal

            x_hat_2 = {}
            for j in employees:
                result = get_employee_tasks(
                    j, company_tasks, model, score, story_points, max_employee_workload
                )
                if len(result[1]) > 0:
                    x_hat_2[j] = result
        else:
            print("No Solution Found!")
            x_hat_2 = {}

        """## 6.3. Show the Solver's Result"""

        # Set display options
        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)

        # Convert dictionary to DataFrame and set 'employee' as index
        global result_2
        result_2 = pd.DataFrame.from_dict(
            x_hat_2,
            orient="index",
            columns=[
                "company",
                "assigned_task",
                "sum_sp",
                "wasted_sp",
                "assessment_score",
            ],
        )
        result_2.index.name = "employee"
        result_2.to_csv("./output_CA/result_2.csv")

        """### 6.3.1 Statistics of The Objective"""

        total_employee = len(employees)
        total_sp = sum(story_points.values())
        total_active_employee = len(set(employee for employee in x_hat_2.keys()))
        total_active_sp = sum(value[2] for value in x_hat_2.values())
        total_idle_employee = total_employee - total_active_employee
        total_wasted_sp = total_sp - total_active_sp

        print(f"Total Employee\t\t\t: {total_employee}")
        print(
            f"Total Active Employee\t\t: {total_active_employee}\t{(total_active_employee/total_employee)*100:.2f}%"
        )
        print(
            f"Total Idle Employee\t\t: {total_idle_employee}\t{(total_idle_employee/total_employee)*100:.2f}%\n"
        )
        print(f"Total Story Points\t\t: {total_sp}")
        print(
            f"Total Active Story Points\t: {total_active_sp}\t{(total_active_sp/total_sp)*100:.2f}%"
        )
        print(
            f"Total Wasted Story Points\t: {total_wasted_sp}\t{(total_wasted_sp/total_sp)*100:.2f}%\n"
        )

        """### 6.3.2. Distribution With Respect to the Assessment Score"""

        # timer for auto close plot
        timer = threading.Timer(3, close_plot)
        timer.start()

        # make boxplot for objective 1 with respect to the assessment score
        assessment_score_2 = (
            result_2["assessment_score"].explode().reset_index(drop=True)
        )

        if len(assessment_score_2) != 0:
            assessment_score_2.plot(kind="box")
            plt.title("Assessment Score Boxplot of Objective 2")
            plt.show()
        else:
            print("No data to show")

        return mu_Z_2, mu_Z_star, assessment_score_2

    except Exception as e:
        print(f"An error occured in s6_objective2: {e}")
        print(f"An error occurred in s6_objective2: {e}")
        return None, None, None


def s7_objective3(
    model,
    employees,
    company_tasks,
    score,
    story_points,
    max_employee_workload,
    max_workload,
    mu_Z_star,
):
    """# 7. Single Objective Approach: 3) Balancing Workload For Each Employee
    ## 7.1. Set The Objective Model

    $$
    \\mu_{Z_3} = min.\\:W_{j} \\quad \\tag{3}
    $$
    """

    try:
        # single objective 3
        mu_Z_3 = max_workload
        model.setObjective(mu_Z_3, GRB.MINIMIZE)

        """## 7.2. Solve The Model of Objective $(3)$"""

        # solve the model
        model.optimize()

        """### 7.2.1 Print The Solver Results"""

        # Check and process the solution
        if model.status == GRB.OPTIMAL:
            print("Solution Found!")
            print(
                f"Obj. Value 3 i.e. Maximum Story Points Each Employee: {model.ObjVal}\n"
            )
            mu_Z_star[3] = model.ObjVal

            x_hat_3 = {}
            for j in employees:
                result = get_employee_tasks(
                    j, company_tasks, model, score, story_points, max_employee_workload
                )
                if len(result[1]) > 0:
                    x_hat_3[j] = result
        else:
            print("No Solution Found!")
            x_hat_3 = {}

        """## 7.3. Show the Solver's Result"""

        # Set display options
        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)

        # Convert dictionary to DataFrame and set 'employee' as index
        global result_3
        result_3 = pd.DataFrame.from_dict(
            x_hat_3,
            orient="index",
            columns=[
                "company",
                "assigned_task",
                "sum_sp",
                "wasted_sp",
                "assessment_score",
            ],
        )
        result_3.index.name = "employee"

        result_3.to_csv("./output_CA/result_3.csv")

        """### 7.3.1 Statistics of The Objective"""

        total_employee = len(employees)
        total_sp = sum(story_points.values())
        total_active_employee = len(set(employee for employee in x_hat_3.keys()))
        total_active_sp = sum(value[2] for value in x_hat_3.values())
        total_idle_employee = total_employee - total_active_employee
        total_wasted_sp = total_sp - total_active_sp

        print(f"Total Employee\t\t\t: {total_employee}")
        print(
            f"Total Active Employee\t\t: {total_active_employee}\t{(total_active_employee/total_employee)*100:.2f}%"
        )
        print(
            f"Total Idle Employee\t\t: {total_idle_employee}\t{(total_idle_employee/total_employee)*100:.2f}%\n"
        )
        print(f"Total Story Points\t\t: {total_sp}")
        print(
            f"Total Active Story Points\t: {total_active_sp}\t{(total_active_sp/total_sp)*100:.2f}%"
        )
        print(
            f"Total Wasted Story Points\t: {total_wasted_sp}\t{(total_wasted_sp/total_sp)*100:.2f}%\n"
        )

        """### 7.3.2. Distribution With Respect to the Assessment Score"""

        # timer for auto close plot
        timer = threading.Timer(3, close_plot)
        timer.start()

        # make boxplot for objective 1 with respect to the assessment score
        assessment_score_3 = (
            result_3["assessment_score"].explode().reset_index(drop=True)
        )

        if len(assessment_score_3) != 0:
            assessment_score_3.plot(kind="box")
            plt.title("Assessment Score Boxplot of Objective 3")
            plt.show()
        else:
            print("No data to show")

        return mu_Z_3, mu_Z_star, assessment_score_3

    except Exception as e:
        print(f"An error occured in s7_objective3: {e}")
        print(f"An error occurred in s7_objective3: {e}")
        return None, None, None


def s8_MOO_1(
    model,
    employees,
    company_tasks,
    score,
    story_points,
    max_employee_workload,
    mu_Z_1,
    mu_Z_2,
    mu_Z_3,
    assessment_score_1,
    assessment_score_2,
    assessment_score_3,
):
    """# 8. Multi-Objective Approach: 1) Weighted Method
    ## 8.1. Set The Objective Model

    $$
    \\mu_{Z_4} = min.\\:M_{1j} = \\alpha \\cdot min. \\: \\mu_{Z_1} + \\beta \\cdot max. \\: \\mu_{Z_2} + \\gamma \\cdot min. \\: \\mu_{Z_3} \\quad \\tag{4}
    $$
    """

    try:
        alpha = 0
        beta = 0.8
        gamma = 0.2

        # MOO method 1
        mu_Z_4 = (alpha * mu_Z_1) + (beta * mu_Z_2) + (gamma * mu_Z_3)
        model.setObjective(mu_Z_4, GRB.MINIMIZE)

        """## 8.2. Solve The Model of Objective $(4)$"""

        # solve the model
        model.optimize()

        """### 8.2.1 Print The Solver Results"""

        # Check and process the solution
        if model.status == GRB.OPTIMAL:
            print("Solution Found!")
            print(f"Obj. Value 4 i.e. Multi Objective Method 1: {model.ObjVal}\n")

            x_hat_4 = {}
            for j in employees:
                result = get_employee_tasks(
                    j, company_tasks, model, score, story_points, max_employee_workload
                )
                if len(result[1]) > 0:
                    x_hat_4[j] = result
        else:
            print("No Solution Found!")
            x_hat_4 = {}

        """## 8.3. Show the Solver's Result"""

        # Set display options
        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)

        # Convert dictionary to DataFrame and set 'employee' as index
        result_4 = pd.DataFrame.from_dict(
            x_hat_4,
            orient="index",
            columns=[
                "company",
                "assigned_task",
                "sum_sp",
                "wasted_sp",
                "assessment_score",
            ],
        )
        result_4.index.name = "employee"

        result_4.to_csv("./output_CA/result_4_MOO_1.csv")

        """### 8.3.1 Statistics of The Objective"""

        total_employee = len(employees)
        total_sp = sum(story_points.values())
        total_active_employee = len(set(employee for employee in x_hat_4.keys()))
        total_active_sp = sum(value[2] for value in x_hat_4.values())
        total_idle_employee = total_employee - total_active_employee
        total_wasted_sp = total_sp - total_active_sp

        print(f"Total Employee\t\t\t: {total_employee}")
        print(
            f"Total Active Employee\t\t: {total_active_employee}\t{(total_active_employee/total_employee)*100:.2f}%"
        )
        print(
            f"Total Idle Employee\t\t: {total_idle_employee}\t{(total_idle_employee/total_employee)*100:.2f}%\n"
        )
        print(f"Total Story Points\t\t: {total_sp}")
        print(
            f"Total Active Story Points\t: {total_active_sp}\t{(total_active_sp/total_sp)*100:.2f}%"
        )
        print(
            f"Total Wasted Story Points\t: {total_wasted_sp}\t{(total_wasted_sp/total_sp)*100:.2f}%\n"
        )

        """### 8.3.2. Distribution With Respect to the Assessment Score"""

        # timer for auto close plot
        timer = threading.Timer(3, close_plot)
        timer.start()

        # make boxplot for objective 1 with respect to the assessment score
        assessment_score_4 = (
            result_4["assessment_score"].explode().reset_index(drop=True)
        )

        if len(assessment_score_4) != 0:
            assessment_score_4.plot(kind="box")
            plt.title("Assessment Score Boxplot of MOO Method 1")
            plt.show()
        else:
            print("No data to show")

        """## 8.4 Comparing MOO Method 3 to Single Objective"""

        # timer for auto close plot
        timer = threading.Timer(3, close_plot)
        timer.start()

        # merge all boxplot in one graph
        plt.figure(figsize=(10, 5))
        plt.boxplot(
            [
                assessment_score_1,
                assessment_score_2,
                assessment_score_3,
                assessment_score_4,
            ],
            labels=[
                "Obj 1: Min Idle Employee",
                "Obj 2: Max Similarity Score",
                "Obj 3: Balancing the Workload",
                "MOO Method 1",
            ],
        )
        plt.title("Overall Assessment Score Boxplot")
        plt.xticks(rotation=15)
        plt.savefig("./output_CA/compare_SO_MOO_1.png")
        plt.show()

        return assessment_score_4

    except Exception as e:
        print(f"An error occured in s8_MOO_1: {e}")
        print(f"An error occurred in s8_MOO_1: {e}")


def s9_MOO_2(
    model,
    employees,
    company_tasks,
    score,
    story_points,
    max_employee_workload,
    mu_Z_1,
    mu_Z_2,
    mu_Z_3,
    mu_Z_star,
    assessment_score_1,
    assessment_score_2,
    assessment_score_3,
):
    """# 9. Multi-Objective Approach: 2) Goal Programming Optimization Method
    ## 9.1. Set The Objective Model

    $$
    D = \\sum_{k=1}^{3} \\left( W_{plus_k} \\cdot d_{plus_k} + W_{minus_k} \\cdot d_{minus_k} \\right) / \\mu_{Z_star_k}
    $$
    """

    try:
        # Define weight dictionary for each objective
        Weight = {
            1: 0.05,
            2: 0.85,
            3: 0.1,
        }

        # Define the deviation plus and minus variables
        d_plus = {}
        d_minus = {}

        # Add variables for d_plus and d_minus with specific conditions
        for i in range(1, 4):
            if i != 3:
                d_plus[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"d_plus_{i}")
            if i != 1:
                d_minus[i] = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, name=f"d_minus_{i}"
                )

        # Set specific variables to zero
        d_minus[1] = 0
        d_plus[3] = 0

        mu_Z = {1: mu_Z_1, 2: mu_Z_2, 3: mu_Z_3}

        mu_Z_star_obj = mu_Z_star.copy()
        for i, value in mu_Z_star_obj.items():
            mu_Z_star_obj[i] = 1 / value if value != 0 else 0

        # This constraint is bound for each objective connected with each other
        for k, w in Weight.items():
            if w != 0:
                model.addConstr(mu_Z[k] - d_plus[k] + d_minus[k] == mu_Z_star[k])

        # Define D = sum k=1 to 3 ((W_plus_k * d_plus_k) + (W_minus_k * d_minus_k)) / mu_Z_star iterate
        D = quicksum(
            (
                (Weight[i] * (d_plus[i] + d_minus[i])) * mu_Z_star_obj[i]
                for i in range(1, 4)
            )
        )

        # Minimize D
        model.setObjective(D, GRB.MINIMIZE)

        ## 9.2. Solve The Model

        # Solve the model
        model.optimize()

        ## 9.3. Print The Solver Results

        # Check and process the solution
        if model.status == GRB.OPTIMAL:
            print("Solution Found!")
            print(f"Obj. Value 5 i.e. Deviation: {model.ObjVal}\n")

            x_hat_5 = {}
            for j in employees:
                result = get_employee_tasks(
                    j, company_tasks, model, score, story_points, max_employee_workload
                )
                if len(result[1]) > 0:
                    x_hat_5[j] = result
        else:
            print("No Solution Found!")
            x_hat_5 = {}

        ## 9.4. Show the Solver's Result

        # Set display options
        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)

        # Show data that has positive metrics score
        global result_5
        result_5 = pd.DataFrame.from_dict(
            x_hat_5,
            orient="index",
            columns=[
                "company",
                "assigned_task",
                "sum_sp",
                "wasted_sp",
                "assessment_score",
            ],
        )

        result_5.index.name = "employee"
        result_5.to_csv("./output_CA/result_5_MOO_2.csv")

        ## 9.5. Statistics of The Objective

        total_employee = len(employees)
        total_sp = sum(story_points.values())
        total_active_employee = len(set(employee for employee in x_hat_5.keys()))
        total_active_sp = sum(value[2] for value in x_hat_5.values())
        total_idle_employee = total_employee - total_active_employee
        total_wasted_sp = total_sp - total_active_sp

        print(f"Total Employee\t\t\t: {total_employee}")
        print(
            f"Total Active Employee\t\t: {total_active_employee}\t{(total_active_employee/total_employee)*100:.2f}%"
        )
        print(
            f"Total Idle Employee\t\t: {total_idle_employee}\t{(total_idle_employee/total_employee)*100:.2f}%\n"
        )
        print(f"Total Story Points\t\t: {total_sp}")
        print(
            f"Total Active Story Points\t: {total_active_sp}\t{(total_active_sp/total_sp)*100:.2f}%"
        )
        print(
            f"Total Wasted Story Points\t: {total_wasted_sp}\t{(total_wasted_sp/total_sp)*100:.2f}%\n"
        )

        ## 9.6. Distribution With Respect to the Assessment Score

        # Timer for auto close plot
        timer = threading.Timer(3, close_plot)
        timer.start()

        # Make boxplot for x_hat_5
        assessment_score_5 = (
            result_5["assessment_score"].explode().reset_index(drop=True)
        )

        if len(assessment_score_5) != 0:
            assessment_score_5.plot(kind="box")
            plt.title("Assessment Score Boxplot of MOO Method 2")
            plt.show()
        else:
            print("No data to show")

        ## 9.7. Comparing MOO Method 2 to Single Objective and MOO Method 1

        # Timer for auto close plot
        timer = threading.Timer(3, close_plot)
        timer.start()

        # Merge all boxplot in one graph
        data = [
            assessment_score_1,
            assessment_score_2,
            assessment_score_3,
            assessment_score_5,
        ]

        plt.figure(figsize=(10, 5))
        plt.boxplot(
            data,
            labels=[
                "Objective 1\nMin Idle Employee",
                "Objective 2\nMax Assessment Score",
                "Objective 3\nBalancing the Workload",
                "MOO Method 2\nGoal Programming",
            ],
        )
        plt.title("Overall Assessment Score Boxplot")
        plt.xticks(rotation=15)
        plt.savefig("./output_CA/compare_SO_MOO_2.png")
        plt.show()

        return assessment_score_5

    except Exception as e:
        print(f"An error occured in s9_MOO_2: {e}")
        print(f"An error occurred in s9_MOO_2: {e}")


def wait_for_y():
    while True:
        user_input = input("Press 'Y' to continue: ")
        if user_input.upper() == "Y":
            print("Continuing process...\n\n")
            break
        else:
            print("Invalid input. Please press 'Y'.")


def close_plot():
    plt.close()


def get_timestamp():
    return datetime.datetime.now().strftime("(%H.%M %d/%m/%Y)")


# Stage 10: Data Visualization
def data_visualization_compare():
    try:

        # Read the CSV file and extract assessment scores
        def read_csv_and_extract_scores(type, file_path):
          df = pd.read_csv(file_path)
          def safe_literal_eval(val):
              try:
                  return ast.literal_eval(val)
              except Exception as e:
                  print(f"Error parsing value: {val} - {str(e)}")
                  return []
          df[type] = df[type].apply(safe_literal_eval)
          scores = df[type].explode().tolist()
          return scores

        # Normalize function for similarity scores
        def normalize_assessment_scores(scores, min_score, max_score, weighted_euclidean=False):
            normalized_scores = []
            for score in scores:
                if weighted_euclidean:
                    normalized_score = 1 - ((score - min_score) / (max_score - min_score))
                else:
                    normalized_score = (score - min_score) / (max_score - min_score)
                normalized_scores.append(normalized_score)
            return normalized_scores

        highest_task = [5] * 65
        highest_emp = [0] * 65
        lowest_task = [0] * 65
        lowest_emp = [0] * 65
        
        worst_score = 40.311288741492746 # euclidean_similarity(lowest_emp, highest_task)
        best_score = 0 # euclidean_similarity(highest_emp, highest_task)
        worst_score_ca = -0.8333333333333331 #taken from passing highest task & emp to the competence assessment module
        best_score_ca = 0.8333333333333331 

        csv_files = ["/content/result_1.csv", "/content/result_2.csv", "/content/result_3.csv", "/content/result_5_MOO_2.csv"]

        employees = pd.read_csv(csv_files[0])['employee'].unique().tolist()
        total_employee = len(employees)
        
        # Your x_hat dictionaries and story_points dictionary should be defined here
        # Example: 
        # x_hat_obj1 = {'employee1': [value1, value2, value3], ...}
        # story_points = {'task1': value, 'task2': value, ...}

        # Calculate idle employees
        total_sp = sum(story_points.values())

        # Objective 1
        total_active_employee_obj1 = len(set(x_hat_obj1.keys()))
        total_active_sp_obj1 = sum(value[2] for value in x_hat_obj1.values())
        total_idle_employee_obj1 = total_employee - total_active_employee_obj1
        total_wasted_sp_obj1 = total_sp - total_active_sp_obj1

        # Objective 2
        total_active_employee_obj2 = len(set(x_hat_obj2.keys()))
        total_active_sp_obj2 = sum(value[2] for value in x_hat_obj2.values())
        total_idle_employee_obj2 = total_employee - total_active_employee_obj2
        total_wasted_sp_obj2 = total_sp - total_active_sp_obj2

        # Objective 3
        total_active_employee_obj3 = len(set(x_hat_obj3.keys()))
        total_active_sp_obj3 = sum(value[2] for value in x_hat_obj3.values())
        total_idle_employee_obj3 = total_employee - total_active_employee_obj3
        total_wasted_sp_obj3 = total_sp - total_active_sp_obj3

        # MOO
        total_active_employee_moo = len(set(x_hat_moo.keys()))
        total_active_sp_moo = sum(value[2] for value in x_hat_moo.values())
        total_idle_employee_moo = total_employee - total_active_employee_moo
        total_wasted_sp_moo = total_sp - total_active_sp_moo

        # Calculate idle employees for assessment scores
        def calculate_idle_employees(file_path):
            df = pd.read_csv(file_path)
            all_employees = df['employee'].unique().tolist()
            active_employees = df[df['assigned_task'] != "[]"]['employee'].unique().tolist()
            idle_employees = len(list(set(all_employees) - set(active_employees)))
            return idle_employees

        idle_employee_obj1_ca = calculate_idle_employees(csv_files[0])
        idle_employee_obj2_ca = calculate_idle_employees(csv_files[1])
        idle_employee_obj3_ca = calculate_idle_employees(csv_files[2])
        idle_employee_obj_moo_ca = calculate_idle_employees(csv_files[3])

        # Bar chart of idle employees
        plt.figure(figsize=(10,5))
        plt.bar(['Obj1 - CA', 'Obj1 - WED', 'Obj2 - CA', 'Obj2 - WED', 'Obj3 - CA', 'Obj3 - WED', 'MOO - CA', 'MOO - WED'],
                [idle_employee_obj1_ca, total_idle_employee_obj1, idle_employee_obj2_ca, total_idle_employee_obj2, idle_employee_obj3_ca, total_idle_employee_obj3, idle_employee_obj_moo_ca, total_idle_employee_moo])
        plt.title('Idle Employees Bar Chart')
        plt.xticks(rotation=15)
        plt.savefig("./output/idle_emp_all.png")
        plt.show()

        # Extract and normalize similarity scores
        ## normalize values of CA
        scores1 = read_csv_and_extract_scores('assessment_score', csv_files[0])
        scores2 = read_csv_and_extract_scores('assessment_score', csv_files[1])
        scores3 = read_csv_and_extract_scores('assessment_score', csv_files[2])
        scores4 = read_csv_and_extract_scores('assessment_score', csv_files[3])

        scores1 = normalize_assessment_scores(scores1, worst_score_ca, best_score_ca, weighted_euclidean=False)
        scores2 = normalize_assessment_scores(scores2, worst_score_ca, best_score_ca, weighted_euclidean=False)
        scores3 = normalize_assessment_scores(scores3, worst_score_ca, best_score_ca, weighted_euclidean=False)
        scores4 = normalize_assessment_scores(scores4, worst_score_ca, best_score_ca, weighted_euclidean=False)

        ## normalize values of WDE
        similarity_score1 = result_obj1['similarity_score'].explode().reset_index(drop=True)
        similarity_score2 = result_obj2['similarity_score'].explode().reset_index(drop=True)
        similarity_score3 = result_obj3['similarity_score'].explode().reset_index(drop=True)
        similarity_score_moo = result_moo['similarity_score'].explode().reset_index(drop=True)

        similarity_score1 = normalize_assessment_scores(similarity_score1, worst_score, best_score, weighted_euclidean=True)
        similarity_score2 = normalize_assessment_scores(similarity_score2, worst_score, best_score, weighted_euclidean=True)
        similarity_score3 = normalize_assessment_scores(similarity_score3, worst_score, best_score, weighted_euclidean=True)
        similarity_score_moo = normalize_assessment_scores(similarity_score_moo, worst_score, best_score, weighted_euclidean=True)

        # timer for auto close plot
        timer = threading.Timer(3, close_plot)
        timer.start()

        # Boxplot for similarity scores
        plt.figure(figsize=(10, 5))
        plt.boxplot([scores1, similarity_score1, scores2, similarity_score2, scores3, similarity_score3, scores4, similarity_score_moo],
                    labels=['Obj1 - CA', 'Obj1 - WED', 'Obj2 - CA', 'Obj2 - WED', 'Obj3 - CA', 'Obj3 - WED', 'MOO - CA', 'MOO - WED'])
        plt.title('Similarity Score Boxplot')
        plt.savefig("./output/similarity_all.png")
        plt.show()

        # Section 10
        workload_1 = result_obj1['sum_sp'].explode().reset_index(drop=True)
        workload_2 = result_obj2['sum_sp'].explode().reset_index(drop=True)
        workload_3 = result_obj3['sum_sp'].explode().reset_index(drop=True)
        workload_moo = result_moo['sum_sp'].explode().reset_index(drop=True)

        workload1 = result_1['sum_sp'].explode().reset_index(drop=True)
        workload2 = result_2['sum_sp'].explode().reset_index(drop=True)
        workload3 = result_3['sum_sp'].explode().reset_index(drop=True)
        workloadmoo = result_5['sum_sp'].explode().reset_index(drop=True)

        # # Plot the boxplots
        plt.figure(figsize=(10, 5))
        plt.boxplot([workload_1, workload1, workload_2, workload2, workload_3, workload3, workload_moo, workloadmoo],
                            labels=['Obj1 - CA', 'Obj1 - WED', 'Obj2 - CA', 'Obj2 - WED', 'Obj3 - CA', 'Obj3 - WED', 'MOO - CA', 'MOO - WED'])
        plt.title('Workload Boxplot')
        plt.savefig("./output/workload_all.png")
        plt.show()

        print("Stage 10: Data Visualization Successful")
    except Exception as e:
        print(f"Stage 10: Data Visualization Failed with error: {str(e)}")
        raise


# Main function to execute all stages
def main():
    try:
        global mu_Z_star
        mu_Z_star = {i: 0.00 for i in range(1, 4)}

        # Data path
        # new_employee_path = '/content/drive/MyDrive/Tadika Mesra Bunga Matahari/#1 Optimization Problem/project1_task-assignment/data/experimental/medium_data_employee.csv'
        # new_task_path = '/content/drive/MyDrive/Tadika Mesra Bunga Matahari/#1 Optimization Problem/project1_task-assignment/data/experimental/medium_data_task.csv'

        new_employee_path = '/content/drive/MyDrive/Tadika Mesra Bunga Matahari/#1 Optimization Problem/project1_task-assignment/data/experimental/mini_data - employee.csv'
        new_task_path = '/content/drive/MyDrive/Tadika Mesra Bunga Matahari/#1 Optimization Problem/project1_task-assignment/data/experimental/mini_data - task.csv'

        

        # Stage 1
        load_and_preprocess_data(new_employee_path, new_task_path)

        # Stage 2
        calculate_similarity_error()

        # Stage 3
        construct_model_build_variables_set_constraints()

        # Stage 4
        objective1_optimization()

        # Stage 5
        objective2_optimization()

        # Stage 6
        objective3_optimization()

        # Stage 7
        moo_optimization()

        # # Stage 7
        # moo_rm_obj1_optimization()

        # # Stage 8
        # moo_rm_obj2_optimization()

        # # Stage 9
        # moo_rm_obj3_optimization()

        # Section 1
        employees, skills_name, tasks, story_points, company_tasks, score = (
            s1_data_structure_CA()
        )

        section_1_msg_1 = f"Section 1: Data Structure Run Successfully"
        section_1_msg_2 = f"score.csv has been saved in the output_CA/score folder.\n\n"
        print(section_1_msg_1)
        print(section_1_msg_2)
        print(section_1_msg_1)
        print(section_1_msg_2)

        # Section 2
        model = s2_construct_model()
        if model:
            print(f"Section 2: Construct Model Run Successfully\n\n")
            print("Section 2: Construct Model Run Successfully")
        else:
            raise Exception("Model construction failed.")

        # Section 3
        x, y, z, max_employee_workload, max_workload = s3_decision_variable(
            model, tasks, employees, company_tasks
        )
        if x and y:
            print(f"Section 3: Build Decision Variable Run Successfully\n\n")
            print(
                "Section 3: Build Decision Variable Run Successfully"
            )
        else:
            raise Exception("Decision variable construction failed.")

        # Section 4
        s4_constraint(
            model, x, y, z, employees, company_tasks, story_points, max_workload
        )
        print(f"Section 4: Set Constraint Run Successfully\n\n")
        print("Section 4: Set Constraint Run Successfully")

        """
        =============================================
        
                    Solve The Objective
        
        =============================================
        """

        mu_Z_star = {i: 0.00 for i in range(1, 4)}

        # Section 5
        print("Section 5: Objective 1 START")
        start_time = datetime.datetime.now()
        mu_Z_1, mu_Z_star, assessment_score_1 = s5_objective1(
            model,
            employees,
            company_tasks,
            y,
            score,
            story_points,
            max_employee_workload,
            mu_Z_star,
        )
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).seconds

        if mu_Z_1 and assessment_score_1 is not None:
            print(f"Section 5: Objective 1 Run Successfully\n\n")
            print(
                f"Section 5: Objective 1 Run Successfully with {duration} seconds"
            )
        else:
            print("Objective 1 failed.")
            raise Exception("Objective 1 failed.")

        # Section 6
        print("Section 6: Objective 2 START")
        start_time = datetime.datetime.now()
        mu_Z_2, mu_Z_star, assessment_score_2 = s6_objective2(
            model,
            employees,
            company_tasks,
            z,
            score,
            story_points,
            max_employee_workload,
            mu_Z_star,
        )
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).seconds

        if mu_Z_2 and assessment_score_2 is not None:
            print(
                f"Section 6: Objective 2 Run Successfully with {duration} seconds"
            )
            print(f"Section 6: Objective 2 Run Successfully\n\n")
        else:
            print("Objective 2 failed.")
            raise Exception("Objective 2 failed.")

        # Section 7
        print("Section 7: Objective 3 START")
        start_time = datetime.datetime.now()
        mu_Z_3, mu_Z_star, assessment_score_3 = s7_objective3(
            model,
            employees,
            company_tasks,
            score,
            story_points,
            max_employee_workload,
            max_workload,
            mu_Z_star,
        )
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).seconds

        if mu_Z_3 and assessment_score_3 is not None:
            print(
                f"Section 7: Objective 3 Run Successfully with {duration} seconds"
            )
            print(f"Section 7: Objective 3 Run Successfully\n\n")
        else:
            print("Objective 3 failed.")
            raise Exception("Objective 3 failed.")

        # Section 8
        # send_discord_notification("Section 8: MOO Method 1 START")
        # start_time = datetime.datetime.now()
        # assessment_score_4 = s8_MOO_1(
        #     model,
        #     employees,
        #     company_tasks,
        #     score,
        #     story_points,
        #     max_employee_workload,
        #     mu_Z_1,
        #     mu_Z_2,
        #     mu_Z_3,
        #     assessment_score_1,
        #     assessment_score_2,
        #     assessment_score_3,
        # )
        # end_time = datetime.datetime.now()
        # duration = (end_time - start_time).seconds

        # if assessment_score_4 is not None:
        #     send_discord_notification(
        #         f"Section 8: MOO Method 1 Run Successfully with {duration} seconds"
        #     )
        #     print(f"Section 8: MOO Method 1 Run Successfully\n\n")
        # else:
        #     send_discord_notification("MOO Method 1 failed.")
        #     raise Exception("MOO Method 1 failed.")

        # Section 9
        print("Section 9: MOO Method 2 START")
        start_time = datetime.datetime.now()
        assessment_score_5 = s9_MOO_2(
            model,
            employees,
            company_tasks,
            score,
            story_points,
            max_employee_workload,
            mu_Z_1,
            mu_Z_2,
            mu_Z_3,
            mu_Z_star,
            assessment_score_1,
            assessment_score_2,
            assessment_score_3,
        )
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).seconds
        if assessment_score_5 is not None:
            print(
                f"Section 9: MOO Method 2 Run Successfully with {duration} seconds"
            )
            print(f"Section 9: MOO Method 2 Run Successfully\n\n")
        else:
            print("MOO Method 2 failed.")
            raise Exception("MOO Method 2 failed.")



        # Stage 10
        data_visualization_compare()

        print("Script ran successfully.")
    except Exception as e:
        error_message = str(e)
        print(f"Script failed with error: {error_message}")

if __name__ == "__main__":
    main()
