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
import pandas as pd

from typing import Dict, List, Tuple, Any

from optimizer import creds, config, helper, create_model, create_objective
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
        mu_Z_1, mu_Z_star, assessment_score_1 = create_objective.s5_objective1(
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
        mu_Z_2, mu_Z_star, assessment_score_2 = create_objective.s6_objective2(
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
        mu_Z_3, mu_Z_star, assessment_score_3 = create_objective.s7_objective3(
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
        assessment_score_4 = create_objective.s8_MOO(
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
