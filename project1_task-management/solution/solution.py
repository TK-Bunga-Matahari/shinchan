"""
Module Name: solution.py
Obective: Task Assignment Optimization Problem Solution

Description:
This module contains the solution for the Task Assignment Optimization Problem.
The solution is divided into several sections, each responsible for a specific task:
1. Data Input: Functions to read and validate input data for the task assignment problem.
2. Optimization Model: Implementation of the optimization model using appropriate algorithms.
3. Constraints and Objectives: Definition of constraints and objective functions used in the optimization.
4. Solution Execution: Functions to execute the optimization model and obtain results.
5. Output Processing: Functions to format and output the results in a user-friendly manner.

Functions:
- s1_data_structure(employee_path, task_path): Pre-process and structure employee and task data.
- s2_construct_model(license_path): Construct the optimization model with specified parameters.
- s3_decision_variable(model, tasks, employees, company_tasks): Define decision variables for the model.
- s4_constraint(model, x, y, z, employees, company_tasks, story_points, max_workload): Set constraints for the optimization model.
- s5_objective1(model, employees, company_tasks, y, score, story_points, max_employee_workload, mu_Z_star): Minimize the idle employee.
- s6_objective2(model, employees, company_tasks, z, score, story_points, max_employee_workload, mu_Z_star): Maximize the assessment score.
- s7_objective3(model, employees, company_tasks, score, story_points, max_employee_workload, max_workload, mu_Z_star): Balance the workload for each employee.
- s8_MOO(model, employees, company_tasks, score, story_points, max_employee_workload, mu_Z_1, mu_Z_2, mu_Z_3, mu_Z_star, assessment_score_1, assessment_score_2, assessment_score_3): Multi-Objective Optimization using Goal Programming.

Classes:
- GapCallback: A callback class to report optimization progress and gap.

Usage:
Import the module in requirement file and yippy file to process the assessment.
The solution can be executed by running the main() function, which orchestrates
the entire workflow from data input to output processing.

Example:
    from task_assignment_optimization import main

    if __name__ == "__main__":
        main()

Author:
TK Bunga Matahari Team
N. Muafi, I.G.P. Wisnu N., F. Zaid N., Fauzi I.S., Joseph C.L., S. Alisya

Last Modified:
June 2024

"""

# Import library
import os
import json
import requests
import datetime
import threading
import pandas as pd
import gurobipy as gp
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from typing import Dict, List, Tuple, Any
from gurobipy import GRB, Model, quicksum
from yippy import CompetencyAssessment


# Load environment variables from .env file
load_dotenv()


def s1_data_structure(
    employee_path: str, task_path: str, overqualification: bool
) -> Tuple[
    List[str],  # employees
    List[str],  # tasks
    Dict[str, int],  # story_points
    Dict[str, List[str]],  # company_tasks
    Dict[str, Dict[str, float]],  # score
    Dict[str, Any],  # info
]:
    """
    Sets up the data structure by processing employee, task data, and calculate skills metric score.

    Args:
        employee_path (str): The path to the employee data CSV file.
        task_path (str): The path to the task data CSV file.
        overqualification (bool): Flag to choice skills metric methodhology.

    Returns:
        Tuple: Contains employees, tasks, story_points, company_tasks, score, and info.

    Dataset Structure:
    The following are examples of how the employee and task datasets must be created:

    Employee Data:
    >>> import pandas as pd
    >>> employee_skills_df = pd.DataFrame({
    ...     'math': [5, 3, 4, 4, 2],
    ...     'python': [5, 4, 4, 4, 3],
    ...     'sql': [3, 5, 4, 5, 2],
    ...     'cloud': [2, 4, 3, 5, 4],
    ...     'database': [2, 3, 4, 5, 4],
    ...     'optimization': [5, 5, 3, 4, 1]
    ... }, index=['Talent 1', 'Talent 2', 'Talent 3', 'Talent 4', 'Talent 5'])
    >>> employee_skills_df
            math  python  sql  cloud  database  optimization
    Talent 1     5       5    3      2         2             5
    Talent 2     3       4    5      4         3             5
    Talent 3     4       4    4      3         4             3
    Talent 4     4       4    5      5         5             4
    Talent 5     2       3    2      4         4             1

    Task Data:
    >>> task_df = pd.DataFrame({
    ...     'project_id': ['P2', 'P2', 'P3', 'P3', 'P2', 'P2', 'P3', 'P1', 'P1', 'P3'],
    ...     'story_points': [1, 2, 3, 4, 0, 0, 0, 5, 2, 5],
    ...     'math': [0, 3, 5, 4, 0, 4, 3, 3, 0, 5],
    ...     'python': [5, 3, 4, 3, 2, 1, 3, 4, 3, 5],
    ...     'sql': [3, 5, 4, 3, 1, 5, 4, 5, 2, 5],
    ...     'cloud': [4, 4, 5, 3, 0, 5, 4, 5, 0, 5],
    ...     'database': [4, 3, 5, 3, 1, 0, 3, 5, 2, 0],
    ...     'optimization': [0, 1, 5, 0, 5, 0, 4, 2, 2, 5]
    ... }, index=['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10'])
    >>> task_df
            project_id  story_points  math  python  sql  cloud  database  optimization
    T1              P2             1     0       5    3      4         4             0
    T2              P2             2     3       3    5      4         3             1
    T3              P3             3     5       4    4      5         5             5
    T4              P3             4     4       3    3      3         3             0
    T5              P2             0     0       2    1      0         1             5
    T6              P2             0     4       1    5      5         0             0
    T7              P3             0     3       3    4      4         3             4
    T8              P1             5     3       4    5      5         5             2
    T9              P1             2     0       3    2      0         2             2
    T10             P3             5     5       5    5      5         0             5

    Example:
    employees, tasks, story_points, company_tasks, score, info = s1_data_structure('employees.csv', 'tasks.csv')
    """

    try:
        # 1.1. Pre-Processing: Employee Data
        # Read data
        employee_skills_df = pd.read_csv(employee_path, index_col="employee_id")
        employee_skills_df.drop(columns=["No", "Role"], inplace=True, errors="ignore")

        employees = employee_skills_df.index.tolist()
        skills_name = employee_skills_df.columns[1:].tolist()

        # 1.2. Pre-Processing: Task Data
        task_df = pd.read_csv(task_path, index_col="task_id")

        tasks = task_df.index.tolist()
        company_names = list(set(task_df["project_id"]))
        story_points = task_df["story_points"].to_dict()

        # 1.3. Group the task data by company/project
        # convert to dictionary each company and its task
        company_tasks = {}

        for company in company_names:
            company_tasks[company] = task_df[
                task_df["project_id"] == company
            ].index.tolist()

        # sort the company tasks from C1 to C5
        company_tasks = dict(sorted(company_tasks.items()))

        # 1.4. Pre-Processing: Skill Metric Score Calculation
        if overqualification:
            """
            # 1.4.1 Pre-Processing: Competency Assessment
            First, create RCD-ACD Dataframe that we get from Task Dataframe for RCD and from Employee Dataframe for ACD.
            """

            # 1.4.2 Required Competence Data
            rcd_df = task_df.drop(columns=["project_id", "story_points"])
            rcd_df = rcd_df.fillna(0)

            # 1.4.3 Acquired Competence Data
            # create a copy of the original DataFrame
            acd_df = employee_skills_df.copy()
            acd_df = acd_df.fillna(0)

            # 1.4.4 Fit the Data for calculate the MSG score
            ca = CompetencyAssessment(rcd_df, acd_df)
            score, info = ca.fit()
        else:
            info = {}
            score = {}

        # Export the score dictionary to CSV
        score_df = pd.DataFrame.from_dict(score, orient="index")
        score_df.to_csv("./output/score.csv")

        return employees, tasks, story_points, company_tasks, score, info

    except Exception as e:
        send_discord_notification(f"An error occured in s1_data_structure_CA: {e}")
        print(f"An error occurred in s1_data_structure_CA: {e}")
        return [], [], {}, {}, {}, {}


def s2_construct_model(license_params: Dict[str, Any]) -> Model:
    """
    Constructs the optimization model.

    Args:
        license_params (Dict[str, Any]): The Gurobi license parameters.

    Returns:
        Model: The constructed optimization model.

    Example:
        model = s2_construct_model(license_params)
    """

    try:
        # Create an environment with WLS license
        env = gp.Env(params=license_params) if license_params else None

        # Create the model within the Gurobi environment
        model = Model(name="task_assignment", env=env)

        # Set Gurobi parameters to improve performance
        model.setParam("Presolve", presolve)  # Aggressive presolve
        model.setParam("MIPFocus", MIPFocus)  # Focus on improving the best bound
        model.setParam("MIPGap", MIPGap)  # 1% optimality gap
        model.setParam("Heuristics", heuristics)  # Increase heuristics effort
        model.setParam("Threads", threads)  # Use threads, adjust based on your CPU

        return model

    except Exception as e:
        send_discord_notification(f"An error occured in s2_construct_model: {e}")
        print(f"An error occurred in s2_construct_model: {e}")
        return model


def s3_decision_variable(
    model: Model,
    tasks: List[str],
    employees: List[str],
    company_tasks: Dict[str, List[str]],
) -> Tuple[
    Dict[Tuple[str, str, str], Any],
    Dict[Tuple[str, str], Any],
    Dict[Tuple[str, str], Any],
    int,
    Any,
]:
    """
    Builds the decision variables for the optimization model.

    Args:
        model (Model): The optimization model.
        tasks (List[int]): List of task IDs.
        employees (List[str]): List of employee IDs.
        company_tasks (Dict[str, List[int]]): Dictionary of company tasks.

    Returns:
        Tuple: Contains decision variables x, y, z, max_employee_workload, and max_workload variable.

    Example:
        x, y, z, max_employee_workload, max_workload = s3_decision_variable(model, tasks, employees, company_tasks)
    """

    try:
        # Decision variable x to represent employee j is assigned to task i in project k
        x = {}
        for k, task in company_tasks.items():
            for i in task:
                for j in employees:
                    x[(i, j, k)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

        # Decision variable y to represent employee j is assigned to project k
        y = {}
        for j in employees:
            for k in company_tasks.keys():
                y[(j, k)] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{k}")

        # Decision variable z to represent task i is assigned to employee j
        z = {}
        for i in tasks:
            for j in employees:
                z[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")

        # Decision variable for max workload that can be assigned
        max_workload = model.addVar(
            vtype=GRB.INTEGER, lb=0, ub=max_employee_workload, name="max_workload"
        )

        # Integrate new variables
        model.update()

        return x, y, z, max_employee_workload, max_workload

    except Exception as e:
        send_discord_notification(f"An error occured in s3_decision_variable: {e}")
        print(f"An error occurred in s3_decision_variable: {e}")
        return {}, {}, {}, 0, None


def s4_constraint(
    model: Model,
    x: Dict[Tuple[str, str, str], Any],
    y: Dict[Tuple[str, str], Any],
    z: Dict[Tuple[str, str], Any],
    employees: List[str],
    company_tasks: Dict[str, List[str]],
    story_points: Dict[str, int],
    max_workload: int,
) -> None:
    """
    Adds constraints to the optimization model.

    Args:
        model (Model): The optimization model.
        x (Dict[Tuple[int, str, str], Any]): Decision variable x.
        y (Dict[Tuple[str, str], Any]): Decision variable y.
        z (Dict[Tuple[int, str], Any]): Decision variable z.
        employees (List[str]): List of employee IDs.
        company_tasks (Dict[str, List[int]]): Dictionary of company tasks.
        story_points (Dict[int, int]): Dictionary of story points for each task.
        max_workload (Any): Decision variable for maximum workload.

    Example:
        s4_constraint(model, x, y, z, employees, company_tasks, story_points, max_workload)
    """

    try:
        # constraint 1: each task assigned to one talent
        for k, task in company_tasks.items():
            for i in task:
                model.addConstr(quicksum(x[(i, j, k)] for j in employees) == 1)

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

        # create constraint 2: each employee can only work on one task
        for j in employees:
            # The sum of y[j][k] for all companies (k) should be <= 1
            model.addConstr(quicksum(y[(j, k)] for k in company_tasks.keys()) <= 1)

        # Constraint 3: Employee workload doesn't exceed the capacity
        for j in employees:
            for k, tasks in company_tasks.items():
                model.addConstr(
                    quicksum(story_points[i] * x[(i, j, k)] for i in tasks)
                    <= max_workload
                )

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
        send_discord_notification(f"An error occured in s4_constraint: {e}")
        print(f"An error occurred in s4_constraint: {e}")


def s5_objective1(
    model: Model,
    employees: List[str],
    company_tasks: Dict[str, List[str]],
    y: Dict[Tuple[str, str], Any],
    score: Dict[str, Dict[str, float]],
    story_points: Dict[str, int],
    max_employee_workload: int,
    mu_Z_star: Dict[int, float],
) -> Tuple[Any, Dict[int, float], pd.Series]:
    """
    Sets the first objective to minimize the number of idle employees and solves the model.

    Args:
        model (Model): The optimization model.
        employees (List[str]): List of employee IDs.
        company_tasks (Dict[str, List[str]]): Dictionary of company tasks.
        y (Dict[Tuple[str, str], Any]): Decision variable y.
        score (Dict[str, Dict[str, float]]): Dictionary of metric scores for each employee-task pair.
        story_points (Dict[str, int]): Dictionary of story points for each task.
        max_employee_workload (int): The maximum workload an employee can handle.
        mu_Z_star (Dict[int, float]): Dictionary to store objective values.

    Returns:
        Tuple: Contains objective value, updated mu_Z_star, and assessment score.

    Example:
        mu_Z_1, mu_Z_star, assessment_score_1 = s5_objective1(model, employees, company_tasks, y, score, story_points, max_employee_workload, mu_Z_star)
    """

    try:
        # single objective 1
        idle = []

        for j in employees:
            idle.append(1 - quicksum(y[j, k] for k in company_tasks.keys()))

        mu_Z_1 = quicksum(idle)
        model.setObjective(mu_Z_1, GRB.MINIMIZE)

        # solve the model
        model.optimize()

        # 5.2.1 Print The Solver Results
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

        # 5.3. Show the Solver's Result
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

        # 5.3.1 Statistics of The Objective
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

        # 5.3.2. Distribution With Respect to the Assessment Score
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
        send_discord_notification(f"An error occured in s5_objective1: {e}")
        print(f"An error occurred in s5_objective1: {e}")
        return None, mu_Z_star, pd.Series()


def s6_objective2(
    model: Model,
    employees: List[str],
    company_tasks: Dict[str, List[str]],
    z: Dict[Tuple[str, str], Any],
    score: Dict[str, Dict[str, float]],
    story_points: Dict[str, int],
    max_employee_workload: int,
    mu_Z_star: Dict[int, float],
) -> Tuple[Any, Dict[int, float], pd.Series]:
    """
    Sets the second objective to maximize the assessment score and solves the model.

    Args:
        model (Model): The optimization model.
        employees (List[str]): List of employee IDs.
        company_tasks (Dict[str, List[str]]): Dictionary of company tasks.
        z (Dict[Tuple[str, str], Any]): Decision variable z.
        score (Dict[str, Dict[str, float]]): List of metric scores for each employee-task pair.
        story_points (Dict[str, int]): List of story points for each task.
        max_employee_workload (int): The maximum workload an employee can handle.
        mu_Z_star (Dict[int, float]): Dictionary to store objective values.

    Returns:
        Tuple: Contains objective value, updated mu_Z_star, and assessment score.

    Example:
        mu_Z_2, mu_Z_star, assessment_score_2 = s6_objective2(model, employees, company_tasks, z, score, story_points, max_employee_workload, mu_Z_star)
    """

    try:
        # single objective 2
        mu_Z_2 = quicksum(
            score[j][i] * z[i, j]
            for k, tasks in company_tasks.items()
            for i in tasks
            for j in employees
        )
        model.setObjective(mu_Z_2, GRB.MAXIMIZE)

        # solve the model
        model.optimize()

        # 6.2.1 Print The Solver Results
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

        # 6.3. Show the Solver's Result
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

        # 6.3.1 Statistics of The Objective
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

        # 6.3.2. Distribution With Respect to the Assessment Score
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
        send_discord_notification(f"An error occured in s6_objective2: {e}")
        print(f"An error occurred in s6_objective2: {e}")
        return None, mu_Z_star, pd.Series()


def s7_objective3(
    model: Model,
    employees: List[str],
    company_tasks: Dict[str, List[str]],
    score: Dict[str, Dict[str, float]],
    story_points: Dict[str, int],
    max_employee_workload: int,
    max_workload: Any,
    mu_Z_star: Dict[int, float],
) -> Tuple[Any, Dict[int, float], pd.Series]:
    """
    Sets the third objective to balance the workload for each employee and solves the model.

    Args:
        model (Model): The optimization model.
        employees (List[str]): List of employee IDs.
        company_tasks (Dict[str, List[str]]): Dictionary of company tasks.
        score (Dict[str, Dict[str, float]]): List of metric scores for each employee-task pair.
        story_points (Dict[str, int]): List of story points for each task.
        max_employee_workload (int): The maximum workload an employee can handle.
        max_workload (Any): Decision variable for maximum workload.
        mu_Z_star (Dict[int, float]): Dictionary to store objective values.

    Returns:
        Tuple: Contains objective value, updated mu_Z_star, and assessment score.

    Example:
        mu_Z_3, mu_Z_star, assessment_score_3 = s7_objective3(model, employees, company_tasks, score, story_points, max_employee_workload, max_workload, mu_Z_star)
    """

    try:
        # single objective 3
        mu_Z_3 = max_workload
        model.setObjective(mu_Z_3, GRB.MINIMIZE)

        # solve the model
        model.optimize()

        # 7.2.1 Print The Solver Results
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

        # 7.3. Show the Solver's Result
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

        # 7.3.1 Statistics of The Objective
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

        # 7.3.2. Distribution With Respect to the Assessment Score
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
        send_discord_notification(f"An error occured in s7_objective3: {e}")
        print(f"An error occurred in s7_objective3: {e}")
        return None, mu_Z_star, pd.Series()


def s8_MOO(
    model: Model,
    employees: List[str],
    company_tasks: Dict[str, List[str]],
    score: Dict[str, Dict[str, float]],
    story_points: Dict[str, int],
    max_employee_workload: int,
    mu_Z_1: Any,
    mu_Z_2: Any,
    mu_Z_3: Any,
    mu_Z_star: Dict[int, float],
    assessment_score_1: Any,
    assessment_score_2: Any,
    assessment_score_3: Any,
) -> Any:
    """
    Sets the multi-objective approach using the Goal Programming Optimization Method and solves the model.

    Args:
        model (Model): The optimization model.
        employees (List[str]): List of employee IDs.
        company_tasks (Dict[str, List[int]]): Dictionary of company tasks.
        score (List[List[float]]): List of metric scores for each employee-task pair.
        story_points (Dict[int, int]): List of story points for each task.
        max_employee_workload (int): The maximum workload an employee can handle.
        mu_Z_1 (Any): Objective value for the first objective.
        mu_Z_2 (Any): Objective value for the second objective.
        mu_Z_3 (Any): Objective value for the third objective.
        mu_Z_star (Dict[int, float]): Dictionary to store objective values.
        assessment_score_1 (Any): Assessment score for the first objective.
        assessment_score_2 (Any): Assessment score for the second objective.
        assessment_score_3 (Any): Assessment score for the third objective.

    Returns:
        Any: Assessment score for the multi-objective approach.

    Example:
        assessment_score_4 = s8_MOO(model, employees, company_tasks, score, story_points, max_employee_workload, mu_Z_1, mu_Z_2, mu_Z_3, mu_Z_star, assessment_score_1, assessment_score_2, assessment_score_3)
    """

    try:
        # define weight dictionary for each objective
        Weight = {
            1: weight_obj1,
            2: weight_obj2,
            3: weight_obj3,
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

        # 8.2. Solve The Model
        gap_callback = GapCallback()
        model.setParam("MIPGap", MIPGap_moo)

        # Solve the model
        model.optimize(gap_callback)

        # 8.3. Print The Solver Results
        # Check and process the solution
        if model.status == GRB.OPTIMAL:
            print("Solution Found!")
            print(f"Obj. Value 5 i.e. Deviation: {model.ObjVal}\n")

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

        # 8.4. Show the Solver's Result
        # Show data that has positive metrics score
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
        result_4.to_csv("./output/result_4_MOO.csv")

        # 8.5. Statistics of The Objective
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

        # 8.6. Distribution With Respect to the Assessment Score
        # Timer for auto close plot
        timer = threading.Timer(3, close_plot)
        timer.start()

        # Make boxplot for x_hat_4
        assessment_score_4 = (
            result_4["assessment_score"].explode().reset_index(drop=True)
        )

        if len(assessment_score_4) != 0:
            assessment_score_4.plot(kind="box")
            plt.title("Assessment Score Boxplot of MOO Method 2")
            plt.show()
        else:
            print("No data to show")

        # 8.7. Comparing MOO Method 2 to Single Objective and MOO Method 1
        # Timer for auto close plot
        timer = threading.Timer(3, close_plot)
        timer.start()

        # Merge all boxplot in one graph
        data = [
            assessment_score_1,
            assessment_score_2,
            assessment_score_3,
            assessment_score_4,
        ]

        plt.figure(figsize=(10, 5))
        plt.boxplot(
            data,
            label=[
                "Objective 1\nMin Idle Employee",
                "Objective 2\nMax Assessment Score",
                "Objective 3\nBalancing the Workload",
                "MOO with\nGoal Programming",
            ],
        )
        plt.title("Overall Assessment Score Boxplot")
        plt.xticks(rotation=15)
        plt.savefig("./output/compare_SO_MOO.png")
        plt.show()

        return assessment_score_4

    except Exception as e:
        send_discord_notification(f"An error occured in s8_MOO: {e}")
        print(f"An error occurred in s8_MOO: {e}")

        return pd.Series()


class GapCallback:
    """
    A callback class for monitoring and reporting the optimization gap during the Mixed Integer Programming (MIP) solving process.

    Attributes:
        reported_gaps (set): A set to keep track of the reported gaps to avoid duplicate notifications.
    """

    def __init__(self) -> None:
        """
        Initializes the GapCallback instance with an empty set for reported gaps.

        Example:
            callback = GapCallback()
        """
        self.reported_gaps = set()

    def __call__(self, model: Model, where: int) -> None:
        """
        The callback function that gets called during the MIP solving process. It monitors the optimization gap and sends notifications when certain conditions are met.

        Args:
            model (gurobipy.Model): The optimization model being solved.
            where (int): An integer code indicating the point in the solving process when the callback is called.

        Example:
            model = Model()
            callback = GapCallback()
            model.optimize(callback)
        """
        if where == GRB.Callback.MIP:
            nodecount = model.cbGet(GRB.Callback.MIP_NODCNT)
            if (
                nodecount % 100 == 0
            ):  # Adjust the frequency of the callback call if needed
                obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
                obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
                if obj_best < GRB.INFINITY and obj_bound > -GRB.INFINITY:
                    gap = abs((obj_bound - obj_best) / obj_best) * 100
                    percentage_gap = gap

                    # Report gap for multiples of 5
                    if percentage_gap > 10 and int(percentage_gap) % 5 == 0:
                        if int(percentage_gap) not in self.reported_gaps:
                            print(f"Model reached {int(percentage_gap)}% gap.")
                            send_discord_notification(
                                f"Model reached {int(percentage_gap)}% gap."
                            )
                            self.reported_gaps.add(int(percentage_gap))

                    # Report gap for each integer when gap <= 10
                    elif percentage_gap <= 10:
                        if int(percentage_gap) not in self.reported_gaps:
                            print(f"Model reached {percentage_gap}% gap.")
                            send_discord_notification(
                                f"Model reached {percentage_gap}% gap."
                            )
                            self.reported_gaps.add(int(percentage_gap))


# Convert string to boolean
def str_to_bool(value: str) -> bool:
    """
    Convert a string representation of truth to a boolean value.

    This function takes a string and converts it to a boolean. The conversion
    is case-insensitive and considers the strings "true", "1", and "yes" as
    True, and all other strings as False.

    Args:
        value (str): The string to be converted to a boolean. Expected values are
                     case-insensitive "true", "1", "yes" for True, and anything else for False.

    Returns:
        bool: The boolean value corresponding to the string representation.
              Returns True for "true", "1", "yes" (case-insensitive), and False otherwise.

    Examples:
        >>> str_to_bool("true")
        True
        >>> str_to_bool("False")
        False
        >>> str_to_bool("YES")
        True
        >>> str_to_bool("no")
        False
    """
    return value.lower() in ("true", "1", "yes")


def get_employee_tasks(
    j: str,
    company_tasks: Dict[str, List[str]],
    model: Model,
    score: Dict[str, Dict[str, float]],
    story_points: Dict[str, int],
    max_employee_workload: int,
) -> Tuple[List[str], List[str], int, int, List[float]]:
    """
    Extracts and prints the tasks assigned to an employee and computes related metrics.

    Args:
        j (int): The employee ID.
        company_tasks (Dict[str, List[int]]): Dictionary of company tasks.
        model (Model): The optimization model.
        score (List[List[float]]): List of metric scores for each employee-task pair.
        story_points (List[int]): List of story points for each task.
        max_employee_workload (int): The maximum workload an employee can handle.

    Returns:
        Tuple[List[str], List[int], int, int, List[float]]: A tuple containing:
            - List of company names (comp)
            - List of task IDs (task)
            - Total story points (sp)
            - Wasted story points (wasted_sp)
            - List of metric scores (sim)

    Example:
        company_tasks = {'CompanyA': [1, 2], 'CompanyB': [3]}
        model = Model()
        score = [[0.5, 0.7, 0.6], [0.4, 0.8, 0.5]]
        story_points = [3, 2, 5]
        max_employee_workload = 10
        get_employee_tasks(1, company_tasks, model, score, story_points, max_employee_workload)
    """
    task = []
    sim = []
    comp = []
    sp = 0

    for k, tasks in company_tasks.items():
        for i in tasks:
            var = model.getVarByName(f"x_{i}_{j}_{k}")
            if var is not None and var.X == 1:
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


def read_license_file(filepath: str) -> Dict[str, str]:
    """
    Reads the Gurobi license file and extracts the license parameters.

    Args:
        filepath (str): The path to the license file.

    Returns:
        Dict[str, str]: A dictionary containing the license parameters.

    Example:
        license_params = read_license_file("gurobi.lic")
    """
    params = {}
    with open(filepath, "r") as file:
        for line in file:
            if line.startswith("WLSACCESSID"):
                params["WLSACCESSID"] = line.split("=")[1].strip()
            elif line.startswith("WLSSECRET"):
                params["WLSSECRET"] = line.split("=")[1].strip()
            elif line.startswith("LICENSEID"):
                params["LICENSEID"] = int(line.split("=")[1].strip())
    return params


def close_plot() -> None:
    """
    Closes the current plot.

    Example:
        close_plot()
    """
    plt.close()


def get_timestamp() -> str:
    """
    Gets the current timestamp in the format (HH.MM DD/MM/YYYY).

    Returns:
        str: The current timestamp.

    Example:
        timestamp = get_timestamp()
    """
    return datetime.datetime.now().strftime("(%H.%M %d/%m/%Y)")


def send_discord_notification(message: str) -> None:
    """
    Sends a notification with the given message to a Discord channel.

    Args:
        message (str): The message to be sent.

    Example:
        send_discord_notification("Model reached 5% gap.")
    """
    url = "https://discord.com/api/webhooks/1245288786024206398/ZQEM6oSRWOYw0DV9_3WUNGYIk7yZQ-M1OdsZU6J3DhUKhZ-qmi8ecqJRAVBRqwpJt0q8"
    data = {"content": f"{get_timestamp()} {message}"}
    # response = requests.post(
    #     url, data=json.dumps(data), headers={"Content-Type": "application/json"}
    # )

    # if response.status_code == 204:
    #     print("Notification sent successfully.")
    # else:
    #     print("Failed to send notification.")


# Get License Information from Environment Variables
wls_access_id = os.getenv("WLSACCESSID")
wls_secret = os.getenv("WLSSECRET")
license_id = os.getenv("LICENSEID")

if wls_access_id is None or wls_secret is None or license_id is None:
    license_params = {}
else:
    license_params = {
        "WLSACCESSID": wls_access_id,
        "WLSSECRET": wls_secret,
        "LICENSEID": int(license_id),
    }

# Optimization Parameters from Environment Variables
presolve = int(os.getenv("PRESOLVE", 2))
MIPFocus = int(os.getenv("MIPFOCUS", 1))
MIPGap = float(os.getenv("MIPGAP", 0.01))
heuristics = float(os.getenv("HEURISTICS", 0.8))
threads = int(os.getenv("THREADS", 2))
MIPGap_moo = float(os.getenv("MIPGAP_MOO", 0.05))

# Objective Weights from Environment Variables
weight_obj1 = float(os.getenv("WEIGHT_OBJ1", 0.03))
weight_obj2 = float(os.getenv("WEIGHT_OBJ2", 0.9))
weight_obj3 = float(os.getenv("WEIGHT_OBJ3", 0.07))

# Methodology from Environment Variables
overqualification = str_to_bool(os.getenv("OVERQUALIFICATION", "True"))

# File Paths from Environment Variables
employee_path = os.getenv("EMPLOYEE_PATH", "./data/employees_data.csv")
task_path = os.getenv("TASK_PATH", "./data/tasks_data.csv")

# Maximum Workload from Environment Variables
max_employee_workload = int(os.getenv("MAX_EMPLOYEE_WORKLOAD", 20))


def main():
    """
    The main function that executes the steps for the task assignment optimization problem.
    """

    header = """
    ==============================================

        TASK ASSIGNMENT OPTIMIZATION PROBLEM

    ==============================================
    """
    print(header)

    header_msg = (
        "Task Assignment Optimization Problem: START with Competence Assessment"
    )
    send_discord_notification(header_msg)

    print("\nExecuting the Steps...\n\n")

    try:
        # Section 1
        employees, tasks, story_points, company_tasks, score, info = s1_data_structure(
            employee_path, task_path, overqualification
        )
        section_1_msg_1 = "Section 1: Data Structure Run Successfully"
        print(section_1_msg_1)
        send_discord_notification(section_1_msg_1)

        # Section 2
        model = s2_construct_model(license_params)
        if model:
            print("Section 2: Construct Model Run Successfully\n\n")
            send_discord_notification("Section 2: Construct Model Run Successfully")
        else:
            raise Exception("Model construction failed.")

        # Section 3
        x, y, z, max_employee_workload, max_workload = s3_decision_variable(
            model, tasks, employees, company_tasks
        )
        if x and y and z:
            print("Section 3: Build Decision Variable Run Successfully\n\n")
            send_discord_notification(
                "Section 3: Build Decision Variable Run Successfully"
            )
        else:
            raise Exception("Decision variable construction failed.")

        # Section 4
        s4_constraint(
            model, x, y, z, employees, company_tasks, story_points, max_workload
        )
        print("Section 4: Set Constraint Run Successfully\n\n")
        send_discord_notification("Section 4: Set Constraint Run Successfully")

        print("\nSolving The Objective...\n\n")

        mu_Z_star = {i: 0.00 for i in range(1, 4)}

        # Section 5
        send_discord_notification("Section 5: Objective 1 START")
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
            print("Section 5: Objective 1 Run Successfully\n\n")
            send_discord_notification(
                f"Section 5: Objective 1 Run Successfully with {duration} seconds"
            )
        else:
            send_discord_notification("Objective 1 failed.")
            raise Exception("Objective 1 failed.")

        # Section 6
        send_discord_notification("Section 6: Objective 2 START")
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
            send_discord_notification(
                f"Section 6: Objective 2 Run Successfully with {duration} seconds"
            )
            print("Section 6: Objective 2 Run Successfully\n\n")
        else:
            send_discord_notification("Objective 2 failed.")
            raise Exception("Objective 2 failed.")

        # Section 7
        send_discord_notification("Section 7: Objective 3 START")
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
            send_discord_notification(
                f"Section 7: Objective 3 Run Successfully with {duration} seconds"
            )
            print("Section 7: Objective 3 Run Successfully\n\n")
        else:
            send_discord_notification("Objective 3 failed.")
            raise Exception("Objective 3 failed.")

        # Section 8
        send_discord_notification("Section 8: MOO START")
        start_time = datetime.datetime.now()
        assessment_score_4 = s8_MOO(
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
        if assessment_score_4 is not None:
            send_discord_notification(
                f"Section 8: MOO Run Successfully with {duration} seconds"
            )
            print("Section 8: MOO Run Successfully\n\n")
        else:
            send_discord_notification("MOO failed.")
            raise Exception("MOO failed.")

    except Exception as e:
        send_discord_notification(f"An error occurred: {e}")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    try:
        main()
        send_discord_notification("Script ran successfully.")
    except Exception as e:
        error_message = str(e)
        send_discord_notification(f"Script failed with error: {error_message}")
