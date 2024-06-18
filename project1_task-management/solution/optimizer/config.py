# import library
import os


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


try:
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

    metrics = (
        "CompetencyAssessment" if overqualification else "WeightedEuclideanDistance"
    )
except Exception as e:
    print(f"Error in getting the Optimization Parameters: {e}")
