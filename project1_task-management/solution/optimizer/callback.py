from optimizer.helper import send_discord_notification
from gurobipy import Model, GRB


class GapCallback:
    """
    A callback class for monitoring and reporting the optimization gap during the Mixed Integer Programming (MIP) solving process.

    Attributes:
        reported_gaps (set): A set to keep track of the reported gaps to avoid duplicate notifications.
    """

    def __init__(self) -> None:
        """
        Initializes the GapCallback instance with an empty set for reported gaps.

        Example:
            callback = GapCallback()
        """
        self.reported_gaps = set()

    def __call__(self, model: Model, where: int) -> None:
        """
        The callback function that gets called during the MIP solving process. It monitors the optimization gap and sends notifications when certain conditions are met.

        Args:
            model (gurobipy.Model): The optimization model being solved.
            where (int): An integer code indicating the point in the solving process when the callback is called.

        Example:
            model = Model()
            callback = GapCallback()
            model.optimize(callback)
        """
        if where == GRB.Callback.MIP:
            nodecount = model.cbGet(GRB.Callback.MIP_NODCNT)
            if (
                nodecount % 100 == 0
            ):  # Adjust the frequency of the callback call if needed
                obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
                obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
                if obj_best < GRB.INFINITY and obj_bound > -GRB.INFINITY:
                    gap = abs((obj_bound - obj_best) / obj_best) * 100
                    percentage_gap = gap

                    # Report gap for multiples of 5
                    if percentage_gap > 10 and int(percentage_gap) % 5 == 0:
                        if int(percentage_gap) not in self.reported_gaps:
                            print(f"Model reached {int(percentage_gap)}% gap.")
                            send_discord_notification(
                                f"Model reached {int(percentage_gap)}% gap."
                            )
                            self.reported_gaps.add(int(percentage_gap))

                    # Report gap for each integer when gap <= 10
                    elif percentage_gap <= 10:
                        if int(percentage_gap) not in self.reported_gaps:
                            print(f"Model reached {percentage_gap}% gap.")
                            send_discord_notification(
                                f"Model reached {percentage_gap}% gap."
                            )
                            self.reported_gaps.add(int(percentage_gap))
