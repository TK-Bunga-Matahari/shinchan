import threading
import pandas as pd
import matplotlib.pyplot as plt
from . import helper
from typing import Dict, List, Tuple, Any
from optimizer.tools import CompetencyAssessment, WeightedEuclideanDistance


def define_data_structure(
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
    employees, tasks, story_points, company_tasks, score, info = define_data_structure('employees.csv', 'tasks.csv')
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


def process_results(
    x_hat: Dict[str, Tuple[str, List[str], int, int, List[float]]],
    employees: List[str],
    story_points: Dict[str, int],
    output_file: str,
    title: str,
    boxplot_title: str,
) -> pd.Series:
    """
    Processes the results, saves the CSV file, shows statistics, and plots the box plot.

    Args:
        x_hat (Dict[str, Tuple[str, List[str], int, int, List[float]]]): Dictionary of results for each employee.
        employees (List[str]): List of employee IDs.
        story_points (Dict[str, int]): Dictionary of story points for each task.
        output_file (str): Path to the output CSV file.
        title (str): Title for the statistics.
        boxplot_title (str): Title for the box plot.

    Returns:
        pd.Series: Series of assessment scores.

    Example:
    >>> assessment_score_1 = preprocessing.process_results(
    ...     x_hat_1,
    ...     employees,
    ...     story_points,
    ...     "./output/result_1.csv",
    ...     "Statistics of Objective 1",
    ...     "Assessment Score Boxplot of Objective 1",
    ... )
    """
    # Set display options
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)

    # Convert dictionary to DataFrame and set 'employee' as index
    result = pd.DataFrame.from_dict(
        x_hat,
        orient="index",
        columns=[
            "company",
            "assigned_task",
            "sum_sp",
            "wasted_sp",
            "assessment_score",
        ],
    )
    result.index.name = "employee"
    result.to_csv(output_file)

    # Statistics of The Objective
    total_employee = len(employees)
    total_sp = sum(story_points.values())
    total_active_employee = len(set(employee for employee in x_hat.keys()))
    total_active_sp = sum(value[2] for value in x_hat.values())
    total_idle_employee = total_employee - total_active_employee
    total_wasted_sp = total_sp - total_active_sp

    print(title)
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

    # Timer for auto close plot
    timer = threading.Timer(3, helper.close_plot)
    timer.start()

    # Make boxplot for the assessment score
    assessment_score = result["assessment_score"].explode().reset_index(drop=True)

    if len(assessment_score) != 0:
        assessment_score.plot(kind="box")
        plt.title(boxplot_title)
        plt.show()
    else:
        print("No data to show")

    return assessment_score