import os
import json
import requests
import datetime
import matplotlib.pyplot as plt

from typing import Dict, Any
from functools import wraps
from . import config


discord_status = False


def start() -> None:
    """
    Startup Information to solve the task assignment optimization problem and make the directory of output.

    Example:
        start()
    """
    header = """
    ==============================================

        TASK ASSIGNMENT OPTIMIZATION PROBLEM

    ==============================================
    """
    print(header)

    header_msg = f"Task Assignment Optimization Problem: START with {config.metrics}"
    show(header_msg, discord_status)

    print("\nExecuting the Steps...\n\n")

    # Define the output directory
    output_directory = "./output"

    # Create the directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)


def notify_and_time(section_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            message = f"{section_name} START"
            show(message, discord_status)

            start_time = datetime.datetime.now()
            try:
                result = func(*args, **kwargs)
                end_time = datetime.datetime.now()
                duration = (end_time - start_time).seconds
                message = f"{section_name} Run Successfully with {duration} seconds"
                show(message, discord_status)

                return result
            except Exception as e:
                message = f"{section_name} failed: {e}"
                show(message, discord_status)
                raise

        return wrapper

    return decorator


def show(msg: Any, status: bool) -> None:
    """
    Show the message into discord and console

    Args:
        msg (Any): The Message that want to show
        status (bool): Discord send message status

    Example:
    >>> show("hello world", True)
    """
    send_discord_notification(msg, status)
    print(msg)


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


def send_discord_notification(message: str, on: bool) -> None:
    """
    Sends a notification with the given message to a Discord channel.

    Args:
        message (str): The message to be sent.
        on (bool): Send Message status is on

    Example:
        send_discord_notification("Model reached 5% gap.")
    """
    # adjust to yours
    if on:
        url = "https://discord.com/api/webhooks/1245288786024206398/ZQEM6oSRWOYw0DV9_3WUNGYIk7yZQ-M1OdsZU6J3DhUKhZ-qmi8ecqJRAVBRqwpJt0q8"
        data = {"content": f"{get_timestamp()} {message}"}
        response = requests.post(
            url, data=json.dumps(data), headers={"Content-Type": "application/json"}
        )

        if response.status_code == 204:
            print("Notification sent successfully.")
        else:
            print("Failed to send notification.")
