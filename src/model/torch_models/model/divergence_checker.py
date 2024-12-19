

class DivergenceChecker:
    """
    A helper class to check for divergence conditions.
    You can customize thresholds or add other conditions as needed.
    """
    def __init__(self, delta_y_threshold=1e2, delta_h_threshold=1e1, logger=print):
        self.delta_y_threshold = delta_y_threshold
        self.delta_h_threshold = delta_h_threshold
        self.logger = logger

    def check(self, delta_y_mask, delta_h):
        """
        Check for divergence conditions based on delta_y_mask and delta_h.

        Parameters
        ----------
        delta_y_mask : torch.Tensor
            The difference in y after last update.
        delta_h : torch.Tensor
            The difference in h after last update.

        Returns
        -------
        bool
            True if divergence is detected, False otherwise.
        """
        if delta_y_mask is not None and delta_y_mask.abs().max() > self.delta_y_threshold:
            self.logger('Flowrate is Diverging')
            return True

        if delta_h is not None and delta_h.abs().max() > self.delta_h_threshold:
            self.logger('H is Diverging')
            return True

        return False
