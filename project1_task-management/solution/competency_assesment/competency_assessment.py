import pandas as pd
import heapq


class CompetencyAssessment:
    def __init__(self, rcd_df, acd_df):
        self.rcd_df = rcd_df
        self.acd_df = acd_df

    def fit(self):
        weight = self.calculate_poc_weight()
        rcd_w, acd_w = self.apply_weight(weight)
        gap = self.cal_gap(rcd_w, acd_w)
        qs = self.calculate_soq_suq(gap)
        qs = self.calculate_MSG(qs)

        info = {"weight": weight, "rcd_w": rcd_w, "acd_w": acd_w, "gap": gap, "qs": qs}

        return qs, info

    def calculate_poc_weight(self):
        required = {}
        weight = {}

        # Store each column (competency) as a separate entry in the 'required' dictionary
        for i, val in self.rcd_df.items():
            required[i] = val

        # Calculate the total required values for each task
        total_req = sum(required.values())

        for competency, val in required.items():
            weight[competency] = {}
            for task, value in val.items():
                if total_req[task] != 0:
                    weight[competency][task] = value / total_req[task]
                else:
                    weight[competency][task] = 0

        return weight

    def apply_weight(self, weight):
        weight_df = pd.DataFrame(weight)

        rcd_w = self.rcd_df.mul(weight_df, axis=1).to_dict("index")

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
                    qs[employee][task] += ["Qualified"]
                else:
                    qs[employee][task] += ["Under-Qualified"]

        return qs

    def rank_MSG(self, qs):
        ranking = {}

        for employee, task_qs in qs.items():
            for task, values in task_qs.items():
                if employee not in ranking:
                    ranking[employee] = {}
                ranking[employee][task] = values[-2]

        for employee in ranking:
            ranking[employee] = dict(
                sorted(
                    ranking[employee].items(), key=lambda item: item[1], reverse=True
                )
            )

        return ranking

    def top_n_score(self, score, n):
        top_n = int(self.rcd_df.shape[0] * (n / 100))
        top_n_score = {}

        for employee, tasks in score.items():
            # Sort tasks by MSG score in descending order and keep only top-n%
            sorted_tasks = dict(
                sorted(tasks.items(), key=lambda item: item[1], reverse=True)[:top_n]
            )
            top_n_score[employee] = sorted_tasks

        return top_n_score

    def top_score(self, score):
        assigned_tasks = {}
        task_assignment = {talent: {} for talent in score}
        all_tasks = {task for tasks in score.values() for task in tasks}
        remaining_tasks = all_tasks.copy()
        talent_heap = []

        # Initialize heap with talents and their max scores
        for talent in score:
            heapq.heappush(talent_heap, (-max(score[talent].values()), talent))

        # Assign tasks based on highest scores first
        while talent_heap and remaining_tasks:
            _, talent = heapq.heappop(talent_heap)
            for task, task_score in sorted(
                score[talent].items(), key=lambda item: item[1], reverse=True
            ):
                if task in remaining_tasks and task_score >= 0:
                    task_assignment[talent][task] = task_score
                    assigned_tasks[task] = (talent, task_score)
                    remaining_tasks.remove(task)
                    break

        # Ensure every talent has at least one task
        for talent in score:
            if not task_assignment[talent]:
                for task, task_score in sorted(
                    score[talent].items(), key=lambda item: item[1], reverse=True
                ):
                    if task not in assigned_tasks:
                        task_assignment[talent][task] = task_score
                        assigned_tasks[task] = (talent, task_score)
                        remaining_tasks.remove(task)
                        break

        # Assign remaining tasks (including those with scores < 0)
        while remaining_tasks:
            for talent in score:
                if not remaining_tasks:
                    break
                for task, task_score in sorted(
                    score[talent].items(), key=lambda item: item[1], reverse=True
                ):
                    if task in remaining_tasks:
                        task_assignment[talent][task] = task_score
                        assigned_tasks[task] = (talent, task_score)
                        remaining_tasks.remove(task)
                        break

        return task_assignment
