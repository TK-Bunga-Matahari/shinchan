import os
import json
import requests
import datetime
import matplotlib.pyplot as plt

from typing import Dict
from . import config, helper


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
    print(header_msg)
    helper.send_discord_notification(header_msg, discord_status)

    print("\nExecuting the Steps...\n\n")

    # Define the output directory
    output_directory = "./output"

    # Create the directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)


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
