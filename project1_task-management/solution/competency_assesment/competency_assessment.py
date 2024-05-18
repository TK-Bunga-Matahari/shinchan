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
      weight[competency] = val/total_req
    
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

    # calculate the msg
    for employee, task_qs in qs.items():
      for task, values in task_qs.items():
        soq, suq = values[0], values[1]
        msg = (soq+suq)/n
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

    # Sort each employee's tasks by MSG in descending order
    for employee in ranking:
        ranking[employee] = dict(sorted(ranking[employee].items(), key=lambda item: item[1], reverse=True))

    return ranking