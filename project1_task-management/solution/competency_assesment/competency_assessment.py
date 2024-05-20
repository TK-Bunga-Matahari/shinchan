import pandas as pd

class CompetencyAssessment():
  def __init__(self, rcd_df, acd_df):
    self.rcd_df = rcd_df
    self.acd_df = acd_df

  def fit(self):
    weight = self.calculate_poc_weight()    
    rcd_w, acd_w = self.apply_weight(weight)
    gap = self.cal_gap(rcd_w, acd_w)
    qs = self.calculate_soq_suq(gap)
    qs = self.calculate_MSG(qs)

    info = {'weight': weight,            
            'rcd_w': rcd_w,
            'acd_w': acd_w,
            'gap': gap,
            'qs': qs
           }

    return qs, info

  def calculate_poc_weight(self):
    required = {}
    weight = {}

    for i, val in self.rcd_df.items():
      required[i] = val

    total_req = sum(required.values())

    for competency, val in required.items():
      weight[competency] = val / total_req

    return weight

  def apply_weight(self, weight):
    weight_df = pd.DataFrame(weight)

    rcd_w = self.rcd_df.mul(weight_df, axis=1).to_dict('index')

    acd_w = {}
    for j, row_j in self.acd_df.iterrows():
      acd_w[j] = {}
      for i, row_i in weight_df.iterrows():
        acd_w[j][i] = row_j * row_i

    return rcd_w, acd_w            

  def cal_gap(self, rcd_w, acd_w):
    competency = self.rcd_df.columns.tolist()

    gap = {employee: {} for employee in self.acd_df.index}

    for employee in acd_w:
      for task in rcd_w:
        for c in competency:
          if task not in gap[employee]:
            gap[employee][task] = {}
          gap[employee][task][c] = acd_w[employee][task][c] - rcd_w[task][c]

    return gap

  def calculate_soq_suq(self, gap):
    qs = {}

    for employee, tasks in gap.items():
      for task, competency in tasks.items():
        soq, suq = 0, 0

        for c, value in competency.items():                    
          if value >= 0:
            soq += value
          elif value < 0:
            suq += value

        if employee not in qs:
          qs[employee] = {}

        qs[employee][task] = [soq, suq]

    return qs

  def calculate_MSG(self, qs):
    n = len(self.acd_df.columns)

    for employee, task_qs in qs.items():
      for task, values in task_qs.items():
        soq, suq = values[0], values[1]
        msg = (soq + suq) / n
        qs[employee][task] += [msg]

        if msg >= 0:
          qs[employee][task] += ['Qualified']
        else:
          qs[employee][task] += ['Under-Qualified']

    return qs

  def rank_MSG(self, qs):
    ranking = {}

    for employee, task_qs in qs.items():
      for task, values in task_qs.items():
        if employee not in ranking:
          ranking[employee] = {}
        ranking[employee][task] = values[-2]

    for employee in ranking:
      ranking[employee] = dict(sorted(ranking[employee].items(), key=lambda item: item[1], reverse=True))

    return ranking
  
  def all_top_n_score(self, score, n):
    top_n = int(self.rcd_df.shape[0] * (n/100))
    top_n_score = {}

    for employee, tasks in score.items():
      # Sort tasks by MSG score in descending order and keep only top-n%
      sorted_tasks = dict(sorted(tasks.items(), key=lambda item: item[1], reverse=True)[:top_n])
      top_n_score[employee] = sorted_tasks
    
    return top_n_score

  def top_n_score(self, score, n):
    # Calculate initial top_n based on the provided percentage
    initial_top_n = int(self.rcd_df.shape[0] * (n / 100))

    # Determine if each employee can get at least 3 tasks
    total_employees = len(score)
    required_tasks = total_employees * 3
    available_tasks = self.rcd_df.shape[0]

    # Adjust top_n if needed
    if initial_top_n * total_employees < required_tasks:
      top_n = available_tasks // total_employees
      print(f"Sorry, for each employee to have 3 tasks, the maximum is only top {top_n * 100 / available_tasks:.2f}%")
    else:
      top_n = initial_top_n

    top_n_score = {}
    assigned_tasks = set()

    sorted_employees = sorted(score.keys(), key=lambda e: max(score[e].values()), reverse=True)

    for employee in sorted_employees:
      sorted_tasks = sorted(score[employee].items(), key=lambda item: item[1], reverse=True)
      top_tasks = []
      for task, task_score in sorted_tasks:
        if len(top_tasks) < top_n and task not in assigned_tasks:
          top_tasks.append((task, task_score))
          assigned_tasks.add(task)
      top_n_score[employee] = dict(top_tasks)

    # Ensure each employee has at least 3 tasks
    for employee in score.keys():
      if len(top_n_score[employee]) < 3:
        remaining_tasks = [(task, score[employee][task]) for task in score[employee] if task not in assigned_tasks]
        remaining_tasks = sorted(remaining_tasks, key=lambda item: item[1], reverse=True)[:3 - len(top_n_score[employee])]
        for task, task_score in remaining_tasks:
          top_n_score[employee][task] = task_score
          assigned_tasks.add(task)

    # Reassign tasks from employees with more than 3 tasks if necessary
    for employee in top_n_score:
      if len(top_n_score[employee]) < 3:
        additional_tasks = []
        for other_employee in top_n_score:
          if other_employee != employee and len(top_n_score[other_employee]) > 3:
            other_tasks = list(top_n_score[other_employee].items())
            for task, task_score in other_tasks:
              if len(top_n_score[employee]) < 3 and task not in top_n_score[employee]:
                additional_tasks.append((task, task_score))
                del top_n_score[other_employee][task]
                if len(top_n_score[other_employee]) <= 3:
                  break
        additional_tasks = sorted(additional_tasks, key=lambda item: item[1], reverse=True)
        while len(top_n_score[employee]) < 3 and additional_tasks:
          task, task_score = additional_tasks.pop(0)
          top_n_score[employee][task] = task_score
          assigned_tasks.add(task)

    return top_n_score
