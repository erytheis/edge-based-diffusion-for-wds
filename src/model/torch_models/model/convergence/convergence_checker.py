from torch_geometric.utils import scatter


class ConvergenceChecker:
    """Base class for convergence criteria."""

    def __init__(self):
        self.current_iteration = 0

    def __iter__(self):
        self.current_iteration = 0
        self.converged = False  # Reset convergence state for a new run
        return self

    def __next__(self):
        # Default: infinite loop unless stopped by subclass logic
        # Raise StopIteration when done
        self.current_iteration += 1
        return self.current_iteration - 1

    def update(self, **kwargs):
        """
        Update the criterion with current iteration metrics.
        This can be overriden by subclasses.
        kwargs could include:
        f_score, h_score, delta_y_mask, delta_h, time, etc.
        """
        pass


class StopPointsConvergenceChecker(ConvergenceChecker):
    """
    Example criterion that stops at a given set of checkpoints
    and checks scores/deltas to decide if convergence or divergence occurs.
    """

    def __init__(self, stop_points, logger=print):
        super().__init__()
        self.stop_points = stop_points
        self.converged = False


    def __next__(self):
        if self.current_iteration >= self.stop_points[-1] or self.converged or self.diverged:
            raise StopIteration
        iteration = self.stop_points[0] + self.current_iteration if self.current_iteration < len(
            self.stop_points) else self.current_iteration
        self.current_iteration += 1
        return iteration

    def update(self,batch, f=None, h=None, delta_y_mask=None, delta_h=None):
        # Check divergence
        if delta_y_mask is not None and delta_y_mask.abs().max() > 1e2:
            self.logger('Flowrate is Diverging')
            self.diverged = True

        if delta_h is not None and delta_h.abs().max() > 1e1:
            self.logger('H is Diverging')
            self.diverged = True

        # Check convergence






