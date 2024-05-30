import numpy as np
import pandas as pd
import math
from gurobipy import Model, GRB, Env, quicksum
import matplotlib.pyplot as plt
import requests
import json
import threading

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

        send_discord_notification("Stage 1: Data Loading & Preprocessing Successful")

    except Exception as e:
        send_discord_notification(f"Stage 1: Data Loading & Preprocessing Failed with error: {str(e)}")
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

        send_discord_notification("Stage 2: Calculating Similarity Error Successful")
        return euclidean_similarity_score
    except Exception as e:
        send_discord_notification(f"Stage 2: Calculating Similarity Error Failed with error: {str(e)}")
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
            model.addConstr(max_workload >= sum(story_points[i] * x[(i, j, k)] for k, tasks in company_tasks.items() for i in tasks), name=f'max_workload_{j}')
        
        send_discord_notification("Weighted Euclidean - Stage 3: Construct Model, Build Variables, Set Constraints Successful")
        # return model, x, y, max_workload, max_employee_workload
    except Exception as e:
        send_discord_notification(f"Weighted Euclidean - Stage 3: Construct Model, Build Variables, Set Constraints Failed with error: {str(e)}")
        raise

# Stage 4: Objective 1 Optimization
def objective1_optimization():
    try:
        I = []
        global x_hat_obj1
        for j in employees:
            obj1 = 1 - sum(y[j, k] for k in company_tasks.keys())
            I.append(obj1)

        global I_total_idle_employee
        I_total_idle_employee = sum(I)
        model.setObjective(I_total_idle_employee, GRB.MINIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            # x_hat_obj1 = get_results(model, employees, company_tasks, "Objective 1")
            # save_x_hat(x_hat_obj1, 'optimization1_results.xlsx')
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

            send_discord_notification("Stage 4: Objective 1 Optimization Successful")
        else:
            send_discord_notification("Stage 4: Objective 1 Optimization No Solution Found!")
            x_hat_obj1 = {}

        # Convert dictionary to DataFrame
        global result_obj1
        result_obj1 = pd.DataFrame([(key, value[0], value[1], value[2], value[3], value[4]) for key, value in x_hat_obj1.items()],
                            columns=['employee', 'company', 'assigned_task', 'sum_sp', 'wasted_sp', 'similarity_score'])


        # Set 'company' as index
        result_obj1.set_index('employee', inplace=True)
        result_obj1.to_csv("./output/result_obj1.csv")


    except Exception as e:
        send_discord_notification(f"Stage 4: Objective 1 Optimization Failed with error: {str(e)}")
        raise


# Stage 5: Objective 2 Optimization
def objective2_optimization():
    try:
        global E_total_similarity_score
        global x_hat_obj2
        E_total_similarity_score = 0
        for k, tasks in company_tasks.items():
            E_total_similarity_score += sum(euclidean_similarity_score[i, j] * x[i, j, k] for i in tasks for j in employees)

        model.setObjective(E_total_similarity_score, GRB.MINIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
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

            send_discord_notification("Stage 5: Objective 2 Optimization Successful")
        else:
            send_discord_notification("Stage 5: Objective 2 Optimization No Solution Found!")
            x_hat_obj2 = {}

        # Convert dictionary to DataFrame
        global result_obj2
        result_obj2 = pd.DataFrame([(key, value[0], value[1], value[2], value[3], value[4]) for key, value in x_hat_obj2.items()],
                            columns=['employee', 'company', 'assigned_task', 'sum_sp', 'wasted_sp', 'similarity_score'])

        # Set 'company' as index
        result_obj2.set_index('employee', inplace=True)
        result_obj2.to_csv("./output/result_obj2.csv")

    except Exception as e:
        send_discord_notification(f"Stage 5: Objective 2 Optimization Failed with error: {str(e)}")
        raise

# Stage 6: Objective 3 Optimization
def objective3_optimization():
    try:
        global x_hat_obj3
        model.setObjective(max_workload, GRB.MINIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
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

            send_discord_notification("Stage 6: Objective 3 Optimization Successful")
        else:
            send_discord_notification("Stage 6: Objective 3 Optimization No Solution Found!")
            x_hat_obj3 = {}

        # Convert dictionary to DataFrame
        global result_obj3
        result_obj3 = pd.DataFrame([(key, value[0], value[1], value[2], value[3], value[4]) for key, value in x_hat_obj3.items()],
                            columns=['employee', 'company', 'assigned_task', 'sum_sp', 'wasted_sp', 'similarity_score'])

        # Set 'company' as index
        result_obj3.set_index('employee', inplace=True)
        result_obj3.to_csv("./output/result_obj3.csv")


    except Exception as e:
        send_discord_notification(f"Stage 6: Objective 3 Optimization Failed with error: {str(e)}")
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

            send_discord_notification("Stage 7: MOO removing obj 1 Optimization Successful")
        else:
            send_discord_notification("Stage 7: MOO removing obj 1 Optimization No Solution Found!")
            x_hat_rm_obj1 = {}

        # Convert dictionary to DataFrame
        global result_rm_obj1
        result_rm_obj1 = pd.DataFrame([(key, value[0], value[1], value[2], value[3], value[4]) for key, value in x_hat_rm_obj1.items()],
                            columns=['employee', 'company', 'assigned_task', 'sum_sp', 'wasted_sp', 'similarity_score'])

        # Set 'company' as index
        result_rm_obj1.set_index('employee', inplace=True)
        result_rm_obj1.to_csv("./output/result_rm_obj1.csv")


    except Exception as e:
        send_discord_notification(f"Stage 7: MOO Removing Objective 1 Optimization Failed with error: {str(e)}")
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

            send_discord_notification("Stage 8: MOO Removing Objective 2 Optimization Successful")
        else:
            send_discord_notification("Stage 8: MOO Removing Objective 2 Optimization No Solution Found!")
            x_hat_rm_obj2 = {}

        # Convert dictionary to DataFrame
        global result_rm_obj2
        result_rm_obj2 = pd.DataFrame([(key, value[0], value[1], value[2], value[3], value[4]) for key, value in x_hat_rm_obj2.items()],
                            columns=['employee', 'company', 'assigned_task', 'sum_sp', 'wasted_sp', 'similarity_score'])

        # Set 'company' as index
        result_rm_obj2.set_index('employee', inplace=True)
        result_rm_obj2.to_csv("./output/result_rm_obj2.csv")

    except Exception as e:
        send_discord_notification(f"Stage 8: MOO Removing Objective 2 Optimization Failed with error: {str(e)}")
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

            send_discord_notification("Stage 8: MOO Removing Objective 2 Optimization Successful")
        else:
            send_discord_notification("Stage 8: MOO Removing Objective 2 Optimization No Solution Found!")
            x_hat_rm_obj3 = {}

        # Convert dictionary to DataFrame
        global result_rm_obj3
        result_rm_obj3 = pd.DataFrame([(key, value[0], value[1], value[2], value[3], value[4]) for key, value in x_hat_rm_obj3.items()],
                            columns=['employee', 'company', 'assigned_task', 'sum_sp', 'wasted_sp', 'similarity_score'])

        # Set 'company' as index
        result_rm_obj3.set_index('employee', inplace=True)
        result_rm_obj3.to_csv("./output/result_rm_obj3.csv")

        
    except Exception as e:
        send_discord_notification(f"Stage 9: MOO Removing Objective 3 Optimization Failed with error: {str(e)}")
        raise

# Stage 10: Data Visualization
def data_visualization():
    try:
        # Calculate idle employees
        total_employee = len(employees)
        total_sp = sum(story_points.values())

        ## Objective 1
        total_active_employee_obj1 = len(set(employee for employee in x_hat_obj1.keys()))
        total_active_sp_obj1 = sum(value[2] for value in x_hat_obj1.values())
        total_idle_employee_obj1 = total_employee - total_active_employee_obj1
        total_wasted_sp_obj1 = total_sp - total_active_sp_obj1

        ## Objective 2
        total_active_employee_obj2 = len(set(employee for employee in x_hat_obj2.keys()))
        total_active_sp_obj2 = sum(value[2] for value in x_hat_obj2.values())
        total_idle_employee_obj2 = total_employee - total_active_employee_obj2
        total_wasted_sp_obj2 = total_sp - total_active_sp_obj2

        ## Objective 3
        total_active_employee_obj3 = len(set(employee for employee in x_hat_obj3.keys()))
        total_active_sp_obj3 = sum(value[2] for value in x_hat_obj3.values())
        total_idle_employee_obj3 = total_employee - total_active_employee_obj3
        total_wasted_sp_obj3 = total_sp - total_active_sp_obj3

        ## Remove Objective 1
        total_active_employee_rm_obj1 = len(set(employee for employee in x_hat_rm_obj1.keys()))
        total_active_sp_rm_obj1 = sum(value[2] for value in x_hat_rm_obj1.values())
        total_idle_employee_rm_obj1 = total_employee - total_active_employee_rm_obj1
        total_wasted_sp_rm_obj1 = total_sp - total_active_sp_rm_obj1

        ## Remove Objective 2
        total_active_employee_rm_obj2 = len(set(employee for employee in x_hat_rm_obj2.keys()))
        total_active_sp_rm_obj2 = sum(value[2] for value in x_hat_rm_obj2.values())
        total_idle_employee_rm_obj2 = total_employee - total_active_employee_rm_obj2
        total_wasted_sp_rm_obj2 = total_sp - total_active_sp_rm_obj2

        ## Remove Objective 3
        total_active_employee_rm_obj3 = len(set(employee for employee in x_hat_rm_obj3.keys()))
        total_active_sp_rm_obj3 = sum(value[2] for value in x_hat_rm_obj3.values())
        total_idle_employee_rm_obj3 = total_employee - total_active_employee_rm_obj3
        total_wasted_sp_rm_obj3 = total_sp - total_active_sp_rm_obj3
        
        # timer for auto close plot
        timer = threading.Timer(3, close_plot)
        timer.start()

        # Create bar chart of idle employees
        plt.figure(figsize=(10,5))
        plt.bar(['Objective 1', 'Objective 2' ,'Objective 3', 'Remove Obj 1', 'Remove Obj 2', 'Remove Obj 3'],
                [total_idle_employee_obj1, total_idle_employee_obj2, total_idle_employee_obj3, total_idle_employee_rm_obj1, total_idle_employee_rm_obj2, total_idle_employee_rm_obj3])
        plt.title('Idle Employees Bar Chart')
        plt.xticks(rotation=15)
        plt.savefig("./output/idle_emp_all.png")
        plt.show()

        # Calculate similarity score
        similarity_score1 = result_obj1['similarity_score'].explode().reset_index(drop=True)
        similarity_score2 = result_obj2['similarity_score'].explode().reset_index(drop=True)
        similarity_score3 = result_obj3['similarity_score'].explode().reset_index(drop=True)
        similarity_score_rm_obj1 = result_rm_obj1['similarity_score'].explode().reset_index(drop=True)
        similarity_score_rm_obj2 = result_rm_obj2['similarity_score'].explode().reset_index(drop=True)
        similarity_score_rm_obj3 = result_rm_obj3['similarity_score'].explode().reset_index(drop=True)

        # timer for auto close plot
        timer = threading.Timer(3, close_plot)
        timer.start()

        plt.figure(figsize=(10, 5))
        plt.boxplot([similarity_score1, similarity_score2, similarity_score3, similarity_score_rm_obj1, similarity_score_rm_obj2, similarity_score_rm_obj3],
                    labels=['Objective 1', 'Objective 2', 'Objective 3', 'Remove Obj 1', 'Remove Obj 2', 'Remove Obj 3'])
        plt.title('Similarity Score Boxplot')
        plt.savefig("./output/similarity_all.png")
        plt.show()


        workload_1 = result_obj1['sum_sp'].explode().reset_index(drop=True)
        workload_2 = result_obj2['sum_sp'].explode().reset_index(drop=True)
        workload_3 = result_obj3['sum_sp'].explode().reset_index(drop=True)
        workload_rm_obj1 = result_rm_obj1['sum_sp'].explode().reset_index(drop=True)
        workload_rm_obj2 = result_rm_obj2['sum_sp'].explode().reset_index(drop=True)
        workload_rm_obj3 = result_rm_obj3['sum_sp'].explode().reset_index(drop=True)

        # timer for auto close plot
        timer = threading.Timer(3, close_plot)
        timer.start()

        plt.figure(figsize=(10, 5))
        plt.boxplot([workload_1, workload_2, workload_3, workload_rm_obj1, workload_rm_obj2, workload_rm_obj3],
                    labels=['Objective 1', 'Objective 2', 'Objective 3', 'Remove Obj 1', 'Remove Obj 2', 'Remove Obj 3'])
        plt.title('Workload Balancing Boxplot')
        plt.savefig("./output/workload_all.png")
        plt.show()

        send_discord_notification("Stage 10: Data Visualization Successful")
    except Exception as e:
        send_discord_notification(f"Stage 10: Data Visualization Failed with error: {str(e)}")
        raise

def close_plot():
    plt.close()


# Main function to execute all stages
def main():
    # Data path
    new_employee_path = './data/fixed_data_employee.csv'
    new_task_path = './data/fixed_data_task.csv'

    try:
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
        moo_rm_obj1_optimization()

        # Stage 8
        moo_rm_obj2_optimization()

        # Stage 9
        moo_rm_obj3_optimization()

        # Stage 10
        data_visualization()

        send_discord_notification("Script ran successfully.")
    except Exception as e:
        error_message = str(e)
        send_discord_notification(f"Script failed with error: {error_message}")

if __name__ == "__main__":
    main()
