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
- s1_data_structure(employee_path, task_path): Pre-process and structure employee, task data, and score metric data.
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

Authors:
TK Bunga Matahari Team
N. Muafi, I.G.P. Wisnu N., F. Zaid N., Fauzi I.S., Joseph C.L., S. Alisya

Last Modified:
June 2024

"""

# Import library
import datetime
import threading
import pandas as pd
import gurobipy as gp
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Any
from gurobipy import GRB, Model, quicksum

from optimizer import creds, config, helper, report, create_model
from optimizer.callback import GapCallback
from optimizer.tools import CompetencyAssessment, WeightedEuclideanDistance


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

        if overqualification:
            # Calculate with Competency Assessment
            ca = CompetencyAssessment(rcd_df, acd_df)
            score, info = ca.fit()
        else:
            # Calculate with Weighted Euclidean Distance
            wed = WeightedEuclideanDistance(rcd_df, acd_df)
            score, info = wed.fit()

        # Export the score dictionary to CSV
        score_df = pd.DataFrame.from_dict(score, orient="index")
        score_df.to_csv("./output/score.csv")

        return employees, tasks, story_points, company_tasks, score, info

    except Exception as e:
        helper.send_discord_notification(
            f"An error occured in s1_data_structure_CA: {e}"
        )
        print(f"An error occurred in s1_data_structure_CA: {e}")
        return [], [], {}, {}, {}, {}


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
                result = report.get_employee_tasks(
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
        timer = threading.Timer(3, helper.close_plot)
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
        helper.send_discord_notification(f"An error occured in s5_objective1: {e}")
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
                result = report.get_employee_tasks(
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
        timer = threading.Timer(3, helper.close_plot)
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
        helper.send_discord_notification(f"An error occured in s6_objective2: {e}")
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
                result = report.get_employee_tasks(
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
        timer = threading.Timer(3, helper.close_plot)
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
        helper.send_discord_notification(f"An error occured in s7_objective3: {e}")
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
            1: config.weight_obj1,
            2: config.weight_obj2,
            3: config.weight_obj3,
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
        model.setParam("MIPGap", config.MIPGap_moo)

        # Solve the model
        model.optimize(gap_callback)

        # 8.3. Print The Solver Results
        # Check and process the solution
        if model.status == GRB.OPTIMAL:
            print("Solution Found!")
            print(f"Obj. Value 5 i.e. Deviation: {model.ObjVal}\n")

            x_hat_4 = {}
            for j in employees:
                result = report.get_employee_tasks(
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
        timer = threading.Timer(3, helper.close_plot)
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

        # 8.7. Comparing MOO to Single Objective
        # Timer for auto close plot
        timer = threading.Timer(3, helper.close_plot)
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
            tick_labels=[
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
        helper.send_discord_notification(f"An error occured in s8_MOO: {e}")
        print(f"An error occurred in s8_MOO: {e}")

        return pd.Series()


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

    header_msg = f"Task Assignment Optimization Problem: START with {config.metrics}"
    print(header_msg)
    helper.send_discord_notification(header_msg)

    print("\nExecuting the Steps...\n\n")

    try:
        # Section 1
        employees, tasks, story_points, company_tasks, score, info = s1_data_structure(
            creds.employee_path, creds.task_path, config.overqualification
        )
        section_1_msg_1 = "Section 1: Data Structure Run Successfully"
        print(section_1_msg_1)
        helper.send_discord_notification(section_1_msg_1)

        # Section 2
        model = create_model.s2_construct_model(creds.license_params)
        if model:
            print("Section 2: Construct Model Run Successfully\n\n")
            helper.send_discord_notification(
                "Section 2: Construct Model Run Successfully"
            )
        else:
            raise Exception("Model construction failed.")

        # Section 3
        x, y, z, max_workload = create_model.s3_decision_variable(
            model, tasks, employees, company_tasks, config.max_employee_workload
        )
        if x and y and z and max_workload:
            print("Section 3: Build Decision Variable Run Successfully\n\n")
            helper.send_discord_notification(
                "Section 3: Build Decision Variable Run Successfully"
            )
        else:
            raise Exception("Decision variable construction failed.")

        # Section 4
        create_model.s4_constraint(
            model,
            x,
            y,
            z,
            max_workload,
            employees,
            company_tasks,
            story_points,
            config.max_employee_workload,
        )
        print("Section 4: Set Constraint Run Successfully\n\n")
        helper.send_discord_notification("Section 4: Set Constraint Run Successfully")

        print("\nSolving The Objective...\n\n")

        mu_Z_star = {i: 0.00 for i in range(1, 4)}

        # Section 5
        helper.send_discord_notification("Section 5: Objective 1 START")
        start_time = datetime.datetime.now()
        mu_Z_1, mu_Z_star, assessment_score_1 = s5_objective1(
            model,
            employees,
            company_tasks,
            y,
            score,
            story_points,
            config.max_employee_workload,
            mu_Z_star,
        )
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).seconds

        if mu_Z_1 and assessment_score_1 is not None:
            print("Section 5: Objective 1 Run Successfully\n\n")
            helper.send_discord_notification(
                f"Section 5: Objective 1 Run Successfully with {duration} seconds"
            )
        else:
            helper.send_discord_notification("Objective 1 failed.")
            raise Exception("Objective 1 failed.")

        # Section 6
        helper.send_discord_notification("Section 6: Objective 2 START")
        start_time = datetime.datetime.now()
        mu_Z_2, mu_Z_star, assessment_score_2 = s6_objective2(
            model,
            employees,
            company_tasks,
            z,
            score,
            story_points,
            config.max_employee_workload,
            mu_Z_star,
        )
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).seconds

        if mu_Z_2 and assessment_score_2 is not None:
            helper.send_discord_notification(
                f"Section 6: Objective 2 Run Successfully with {duration} seconds"
            )
            print("Section 6: Objective 2 Run Successfully\n\n")
        else:
            helper.send_discord_notification("Objective 2 failed.")
            raise Exception("Objective 2 failed.")

        # Section 7
        helper.send_discord_notification("Section 7: Objective 3 START")
        start_time = datetime.datetime.now()
        mu_Z_3, mu_Z_star, assessment_score_3 = s7_objective3(
            model,
            employees,
            company_tasks,
            score,
            story_points,
            config.max_employee_workload,
            max_workload,
            mu_Z_star,
        )
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).seconds

        if mu_Z_3 and assessment_score_3 is not None:
            helper.send_discord_notification(
                f"Section 7: Objective 3 Run Successfully with {duration} seconds"
            )
            print("Section 7: Objective 3 Run Successfully\n\n")
        else:
            helper.send_discord_notification("Objective 3 failed.")
            raise Exception("Objective 3 failed.")

        # Section 8
        helper.send_discord_notification("Section 8: MOO START")
        start_time = datetime.datetime.now()
        assessment_score_4 = s8_MOO(
            model,
            employees,
            company_tasks,
            score,
            story_points,
            config.max_employee_workload,
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
            helper.send_discord_notification(
                f"Section 8: MOO Run Successfully with {duration} seconds"
            )
            print("Section 8: MOO Run Successfully\n\n")
        else:
            helper.send_discord_notification("MOO failed.")
            raise Exception("MOO failed.")

    except Exception as e:
        helper.send_discord_notification(f"An error occurred: {e}")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    try:
        main()
        helper.send_discord_notification("Script ran successfully.")
    except Exception as e:
        error_message = str(e)
        helper.send_discord_notification(f"Script failed with error: {error_message}")
