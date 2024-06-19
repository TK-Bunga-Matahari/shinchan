import threading
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from gurobipy import GRB, Model, quicksum
from . import report, helper, config
from optimizer.callback import GapCallback


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
