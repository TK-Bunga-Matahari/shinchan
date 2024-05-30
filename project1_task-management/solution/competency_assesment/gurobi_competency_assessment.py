"""
# Task Assignment Optimization Problem with Gurobi Framework

_by: TK-Bunga Matahari Team_

---

# 0. The Obligatory Part
"""

# Import library
import json
import requests
import threading
import numpy as np
import pandas as pd
import gurobipy as gp
import matplotlib.pyplot as plt
from gurobipy import GRB, quicksum
from competency_assessment import CompetencyAssessment

EMPLOYEE_PATH = "./mini_data/mini_data - employee.csv"
TASK_PATH = "./mini_data/mini_data - task.csv"


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
        score_df.to_csv("./output/score.csv")

        return employees, skills_name, tasks, story_points, company_tasks, score

    except Exception as e:
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
        WLSACCESSID = "f26730de-b14b-4197-9dbd-7d12372b5d9e"
        WLSSECRET = "4ff3e9d2-037b-47c4-898d-6de34404e994"
        LICENSEID = 2521640

        # Create an environment with WLS license
        params = {
            "WLSACCESSID": WLSACCESSID,
            "WLSSECRET": WLSSECRET,
            "LICENSEID": LICENSEID,
        }
        env = gp.Env(params=params)

        # Create the model within the Gurobi environment
        model = gp.Model(name="task_assignment", env=env)

        return model

    except Exception as e:
        print(f"An error occurred in s2_construct_model: {e}")
        return None


def s3_decision_variable(model, employees, company_tasks):
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
        max_employee_workload = 10

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

        # Decision variable for max workload
        max_workload = model.addVar(
            vtype=GRB.INTEGER, lb=0, ub=max_employee_workload, name="max_workload"
        )

        # Integrate new variables
        model.update()

        return x, y, max_employee_workload, max_workload

    except Exception as e:
        print(f"An error occurred in s3_decision_variable: {e}")
        return {}, {}, 0, None


def s4_constraint(model, x, y, employees, company_tasks, story_points, max_workload):
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

    except Exception as e:
        print(f"An error occurred in s4_constraint: {e}")


def s5_objective1(
    model, employees, company_tasks, y, score, story_points, max_employee_workload
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
        result_1.to_csv("./output/result_1.csv")

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
        assessment_score_1.plot(kind="box")
        plt.title("Assessment Score Boxplot of Objective 1")
        plt.savefig("./output/objective_1.png")
        plt.show()

        return mu_Z_1, assessment_score_1

    except Exception as e:
        print(f"An error occurred in s5_objective1: {e}")
        return None, None


def s6_objective2(
    model, employees, company_tasks, x, score, story_points, max_employee_workload
):
    """# 6. Single Objective Approach: 2) Maximize The Assessment Score
    ## 6.1. Set The Objective Model

    $$
    \\mu _{Z_2} = max.\\:A_{ij} = \\sum _{i\\in \\:I} \\sum _{j\\in \\:J} \\sum _{k\\in \\:K} \\: e_{ij} \\cdot x_{ijk} \\quad \\tag{2}
    $$
    """

    try:
        # objective 2
        assessment_score = []
        assessment_score.append(
            quicksum(
                score[j][i] * x[i, j, k]
                for k, tasks in company_tasks.items()
                for j in employees
                for i in tasks
            )
        )

        mu_Z_2 = quicksum(assessment_score)

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
        result_2.to_csv("./output/result_2.csv")

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
        assessment_score_2.plot(kind="box")
        plt.title("Assessment Score Boxplot of Objective 2")
        plt.savefig("./output/objective_2.png")
        plt.show()

        return mu_Z_2, assessment_score_2

    except Exception as e:
        print(f"An error occurred in s6_objective2: {e}")
        return None, None


def s7_objective3(
    model,
    employees,
    company_tasks,
    score,
    story_points,
    max_employee_workload,
    max_workload,
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

        result_3.to_csv("./output/result_3.csv")

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
        assessment_score_3.plot(kind="box")
        plt.title("Assessment Score Boxplot of Objective 3")
        plt.savefig("./output/objective_3.png")
        plt.show()

        return mu_Z_3, assessment_score_3

    except Exception as e:
        print(f"An error occurred in s7_objective3: {e}")
        return None, None


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
        alpha = 0.1
        beta = 0.8
        gamma = 0.1

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

        result_4.to_csv("./output/result_4_MOO_1.csv")

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
        assessment_score_4.plot(kind="box")
        plt.title("Assessment Score Boxplot of MOO Method 1")
        plt.savefig("./output/MOO_1.png")
        plt.show()

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
        plt.savefig("./output/compare_SO_MOO_1.png")
        plt.show()
        plt.close()

    except Exception as e:
        print(f"An error occurred in s8_MOO_1: {e}")


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


def main():
    header = """
    ==============================================

        TASK ASSIGNMENT OPTIMIZATION PROBLEM

    ==============================================
    """
    print(header)
    wait_for_y()

    header_msg = (
        f"Task Assignment Optimization Problem: START with Competence Assessment"
    )
    send_discord_notification(header_msg)

    """
    =============================================
    
                Execute the Steps
    
    =============================================
    """

    try:
        # Section 1
        employees, skills_name, tasks, story_points, company_tasks, score = (
            s1_data_structure_CA()
        )

        section_1_msg_1 = f"Section 1: Data Structure Run Successfully"
        section_1_msg_2 = f"score.csv has been saved in the output/score folder.\n\n"
        print(section_1_msg_1)
        print(section_1_msg_2)
        send_discord_notification(section_1_msg_1)
        send_discord_notification(section_1_msg_2)

        # Section 2
        model = s2_construct_model()
        if model:
            print(f"Section 2: Construct Model Run Successfully\n\n")
            send_discord_notification("Section 2: Construct Model Run Successfully")
        else:
            raise Exception("Model construction failed.")

        # Section 3
        x, y, max_employee_workload, max_workload = s3_decision_variable(
            model, employees, company_tasks
        )
        if x and y:
            print(f"Section 3: Build Decision Variable Run Successfully\n\n")
            send_discord_notification(
                "Section 3: Build Decision Variable Run Successfully"
            )
        else:
            raise Exception("Decision variable construction failed.")

        # Section 4
        s4_constraint(model, x, y, employees, company_tasks, story_points, max_workload)
        print(f"Section 4: Set Constraint Run Successfully\n\n")
        send_discord_notification("Section 4: Set Constraint Run Successfully")

        """
        =============================================
        
                    Solve The Objective
        
        =============================================
        """

        # Section 5
        mu_Z_1, assessment_score_1 = s5_objective1(
            model,
            employees,
            company_tasks,
            y,
            score,
            story_points,
            max_employee_workload,
        )
        if mu_Z_1 and assessment_score_1 is not None:
            print(f"Section 5: Objective 1 Run Successfully\n\n")
            send_discord_notification("Section 5: Objective 1 Run Successfully")
        else:
            raise Exception("Objective 1 failed.")

        # Section 6
        mu_Z_2, assessment_score_2 = s6_objective2(
            model,
            employees,
            company_tasks,
            x,
            score,
            story_points,
            max_employee_workload,
        )
        if mu_Z_2 and assessment_score_2 is not None:
            print(f"Section 6: Objective 2 Run Successfully\n\n")
            send_discord_notification("Section 6: Objective 2 Run Successfully")
        else:
            raise Exception("Objective 2 failed.")

        # Section 7
        mu_Z_3, assessment_score_3 = s7_objective3(
            model,
            employees,
            company_tasks,
            score,
            story_points,
            max_employee_workload,
            max_workload,
        )
        if mu_Z_3 and assessment_score_3 is not None:
            print(f"Section 7: Objective 3 Run Successfully\n\n")
            send_discord_notification("Section 7: Objective 3 Run Successfully")
        else:
            raise Exception("Objective 3 failed.")

        # Section 8
        s8_MOO_1(
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
        )
        print(f"Section 8: MOO Method 1 Run Successfully\n\n")
        send_discord_notification("Section 8: MOO Method 1 Run Successfully")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    try:
        main()
        send_discord_notification("Script ran successfully.")
    except Exception as e:
        error_message = str(e)
        send_discord_notification(f"Script failed with error: {error_message}")
