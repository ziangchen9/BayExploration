from botorch.acquisition import qKnowledgeGradient

from core.optimizer.acquisition_function.base_acquisition_function import (
    BaseAcquisitionFunction,
)
from core.optimizer.gaussian_model.single_guassian import BaseGPModel


class QKGAcquisitionFunction(BaseAcquisitionFunction):
    def _setup_acquisition_function(self, pg: BaseGPModel, **kwargs):
        num_fantasies = kwargs.get("num_fantasies", 128)
        return qKnowledgeGradient(model=pg.model, num_fantasies=num_fantasies)
