from gurobipy import Model
from typing import Dict, List, Tuple, Any


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
