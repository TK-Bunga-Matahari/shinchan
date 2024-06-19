"""
Module Name: main.py
Obective: Task Assignment Optimization Problem Solution Solver

Description:
This module contains the solution for the Task Assignment Optimization Problem.
The solution is divided into several sections, each responsible for a specific task:
1. Data Input: Functions to read and validate input data for the task assignment problem.
2. Optimization Model: Implementation of the optimization model using appropriate algorithms.
3. Constraints and Objectives: Definition of constraints and objective functions used in the optimization.
4. Solution Execution: Functions to execute the optimization model and obtain results.
5. Output Processing: Functions to format and output the results in a user-friendly manner.

Functions:
- define_data_structure(employee_path, task_path): Pre-process and structure employee, task data, and score metric data.
- construct_model(license_path): Construct the optimization model with specified parameters.
- decision_variables(model, tasks, employees, company_tasks): Define decision variables for the model.
- constraints(model, x, y, z, employees, company_tasks, story_points, max_workload): Set constraints for the optimization model.
- objective1(model, employees, company_tasks, y, score, story_points, max_employee_workload, mu_Z_star): Minimize the idle employee.
- objective2(model, employees, company_tasks, z, score, story_points, max_employee_workload, mu_Z_star): Maximize the assessment score.
- objective3(model, employees, company_tasks, score, story_points, max_employee_workload, max_workload, mu_Z_star): Balance the workload for each employee.
- MOO(model, employees, company_tasks, score, story_points, max_employee_workload, mu_Z_1, mu_Z_2, mu_Z_3, mu_Z_star, assessment_score_1, assessment_score_2, assessment_score_3): Multi-Objective Optimization using Goal Programming.

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
from optimizer import (
    creds,
    config,
    helper,
    preprocessing,
    create_model,
    create_objective,
)


def main():
    """
    The main function that executes the steps for the task assignment optimization problem.
    """

    helper.start()

    try:
        # Section 1
        employees, tasks, story_points, company_tasks, score, info = (
            preprocessing.define_data_structure(
                creds.employee_path, creds.task_path, config.overqualification
            )
        )
        section_1_msg_1 = "Section 1: Define Data Structure Run Successfully"
        print(section_1_msg_1)
        helper.send_discord_notification(section_1_msg_1)

        # Section 2
        model = create_model.construct_model(creds.license_params)
        if model:
            print("Section 2: Construct Model Run Successfully\n\n")
            helper.send_discord_notification(
                "Section 2: Construct Model Run Successfully"
            )
        else:
            raise Exception("Model construction failed.")

        # Section 3
        x, y, z, max_workload = create_model.decision_variables(
            model, tasks, employees, company_tasks, config.max_employee_workload
        )
        if x and y and z and max_workload:
            print("Section 3: Build Decision Variables Run Successfully\n\n")
            helper.send_discord_notification(
                "Section 3: Build Decision Variables Run Successfully"
            )
        else:
            raise Exception("Decision variables construction failed.")

        # Section 4
        create_model.constraints(
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
        print("Section 4: Set Constraints Run Successfully\n\n")
        helper.send_discord_notification("Section 4: Set Constraints Run Successfully")

        print("\nSolving The Objective...\n\n")

        mu_Z_star = {i: 0.00 for i in range(1, 4)}

        # Section 5
        helper.send_discord_notification("Section 5: Objective 1 START")
        start_time = datetime.datetime.now()
        mu_Z_1, mu_Z_star, assessment_score_1 = create_objective.objective1(
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
        mu_Z_2, mu_Z_star, assessment_score_2 = create_objective.objective2(
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
        mu_Z_3, mu_Z_star, assessment_score_3 = create_objective.objective3(
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
        assessment_score_4 = create_objective.MOO(
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
