# TK Bunga Matahari Team Projects

This repository contains multiple optimization projects and challenges developed by the TK Bunga Matahari Team.

## Project Description

This repository is a collection of various challenges and projects created by the TK Bunga Matahari Team. It includes tasks related to course selection, stock selection, a burrito optimization game, a task management project, and optimization toolbox webapp.

## Repository Structure

- `challenge1_course-selection/`: Challenge 1: Course Selection Optimization
- `challenge2_stock-selection/`: Challenge 2: Stock Selection for Portfolio Optimization
- `games1_burrito-optimization-game/`: Burrito Optimization Game
- `project1_task-management/`: Project 1: Scrum Task Assignment Optimization Problem: MOO with Goal Programming Approach
- `web-app/`: Web app files
- `.gitignore`: Git ignore file for ignoring unnecessary files
- `LICENSE`: Initial commit license
- `requirements.in`: Requirements for the project
- `requirements.txt`: Requirements for the project

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. It's recommended to use a virtual environment to manage dependencies.

### Setting Up Virtual Environment

1. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:

   - **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```
   - **Linux/MacOS**:
     ```bash
     source venv/bin/activate
     ```

3. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```

### Setting Up Environment Variables

Create a `.env` file in the `project1_task-management/solution` directory with the necessary environment variables. An example `.env` file might look like this:

```env
# Gurobi License Information
WLSACCESSID=your_wls_access_id
WLSSECRET=your_wls_secret
LICENSEID=your_license_id

# File Paths
EMPLOYEE_PATH=./your/path/to/employees_data.csv
TASK_PATH=./your/path/to/tasks_data.csv
```

Feel free to adjust the `.env` file content example to match the actual environment variables needed for your conditions.

## Usage

Provide examples of how to run the projects and challenges in this repository. For example:

```bash
# Running Challenge 1 Course Selection
python challenge1_course-selection/Solution/course_selection.py

# Running Challenge 2 Stock Selection
python challenge2_stock-selection/solution.py

# Running the Burrito Optimization Game
python games1_burrito-optimization-game/solution.py

# Running the Task Management Project
cd project1_task-management/solution
python solution.py
```

## Authors

TK Bunga Matahari Team <br>
N. Muafi, I.G.P. Wisnu N., F. Zaid N., Fauzi I.S., Joseph C.L., S. Alisya

**Supervisor:**
Yusuf F., Muhajir A.H., Alva A.S.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
