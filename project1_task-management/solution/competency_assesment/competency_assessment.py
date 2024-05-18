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

    rcd_w = self.rcd_df.mul(weight_df, axis=1)
    acd_w = self.acd_df.applymap(lambda x: x * weight_df)
    # acd_w = acd_w.to_dict(orient='index')
        
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
          for v in value:
            if v >= 0:
              soq += v
            elif v < 0:
              suq += v

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