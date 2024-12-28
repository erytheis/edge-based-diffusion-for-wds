import operator

from line_profiler_pycharm import profile

from src.model.torch_models.model.convergence.convergence_condition import \
    EnergyConservationThreshold, MassConservationThreshold, CompositeCondition


class ConvergenceChecker:
    """Base class for convergence criteria."""

    def __init__(self):
        self.current_iteration = 0

    def __iter__(self):
        self.current_iteration = 0
        self.converged = False  # Reset convergence state for a new run
        self.diverged = False
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
        self.logger = logger
        # Create two conditions: one for upper and one for lower conservation
        energy_conservation_condition = EnergyConservationThreshold(threshold=1e-1, norm='mae')
        mass_conservation_condition = MassConservationThreshold(threshold=1e-3, norm='mae')

        # Combine both conditions into a composite condition that requires both to be met
        self.conditions = CompositeCondition(energy_conservation_condition, mass_conservation_condition, operator.and_)

        # Now, you can evaluate combined_condition in your convergence checker

    def __next__(self):
        if self.current_iteration >= self.stop_points[-1] or self.converged or self.diverged:
            raise StopIteration
        iteration = self.stop_points[0] + self.current_iteration if self.current_iteration < len(
            self.stop_points) else self.current_iteration
        self.current_iteration += 1
        return iteration

    @profile
    def update(self, i,  **kwargs):
        if i in self.stop_points:

            # Check divergence
            if kwargs.get('delta_f') is not None and kwargs.get('delta_f').abs().max() > 1e2:
                self.logger('Flowrate is Diverging')
                self.diverged = True

            if kwargs.get('delta_h') is not None and kwargs.get('delta_h').abs().max() > 1e1:
                self.logger('H is Diverging')
                self.diverged = True

            # Check convergence
            if not self.conditions.evaluate(**kwargs):
                # Optional: log which condition failed
                self.logger(f"Conditions are not met.")
                return False
            self.logger(f"All convergence conditions met at iterations {i}.")
            self.converged = True
