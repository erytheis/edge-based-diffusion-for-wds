import logging
import operator
from abc import ABC, abstractmethod

import torch


from src.utils.torch.torch_utils import sparse_agg

logger = logging.getLogger("my_convergence_logger")


class BaseCondition(ABC):
    """
    Abstract base class for convergence conditions.
    Each condition must implement the `evaluate` method
    which returns True if the condition is met, False otherwise.
    """

    def __init__(self, name=None, logger=None):
        self.name = name if name else self.__class__.__name__

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    def __and__(self, other):
        return CompositeCondition(
            self, other, operator.and_, name=f"({self.name} AND {other.name})"
        )

    def __or__(self, other):
        return CompositeCondition(
            self, other, operator.or_, name=f"({self.name} OR {other.name})"
        )


class CompositeCondition(BaseCondition):
    """
    A condition that combines two other conditions with a logical operator.
    """

    def __init__(self, left, right, op, name=None):
        super().__init__(name=name)
        self.left = left
        self.right = right
        self.op = op

    def evaluate(self, **kwargs):
        left_result = self.left.evaluate(**kwargs)
        right_result = self.right.evaluate(**kwargs)

        return self.op(left_result, right_result)


class ConservationBaseCondition(BaseCondition):
    """
    Base condition for checking conservation thresholds (upper or lower).
    Subclasses must define how to extract index and weight from the batch.
    """

    def __init__(self, threshold, norm="mae", name=None):
        super().__init__(name=name)
        self.threshold = threshold
        self.norm = norm

    @abstractmethod
    def get_index_and_weight(self, batch):
        """
        Subclasses must implement this to return the index and weight tensors.
        """
        pass

    @abstractmethod
    def get_x(self, batch):
        """
        Subclasses must implement this to return the index and weight tensors.
        """
        pass

    def get_virtual(self, **kwargs):
        pass

    #@profile
    def evaluate(self, **kwargs):
        batch = kwargs.get("batch")
        x = self.get_x(**kwargs)
        virtual = self.get_virtual(**kwargs)

        if batch is None:
            return False

        index, weight = self.get_index_and_weight(batch)
        eps = sparse_agg(x, index, weight)

        # remove virtual elements with known values, e.g. virtual edges
        if virtual is not None:
            eps =  eps[~virtual]

        if self.norm == "max":
            val = eps.abs().max()
        elif self.norm == "norm":
            val = torch.norm(eps, p=2)
        elif self.norm == "mae":
            val = eps.abs().mean()
        else:
            # Default to mae if unknown norm is provided
            val = eps.abs().mean()

        logger.debug(
            f"[{self.name}] Epsilon result: {val}, Threshold: {self.threshold}"
        )

        return val < self.threshold


class EnergyConservationThreshold(ConservationBaseCondition):
    """
    Checks conservation on the upper Laplacian.
    """
    #@profile
    def get_index_and_weight(self, batch):
        return batch.upper_boundary_index.flip(0), batch.upper_boundary_weight

    def get_x(self, **kwargs):
        return kwargs.get("h")



class MassConservationThreshold(ConservationBaseCondition):
    """
    Checks conservation on the lower Laplacian.
    """

    #@profile
    def get_index_and_weight(self, batch):
        return batch.lower_boundary_index, batch.lower_boundary_weight

    def get_x(self, **kwargs):
        return kwargs.get("f")

    def get_virtual(self, **kwargs):
        return kwargs.get("virtual_nodes")
