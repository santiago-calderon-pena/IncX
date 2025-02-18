from incx.explainers.explainer_enum import ExplainerEnum
from incx.explainers.d_rise import DRise
from incx.explainers.base_explainer import BaseExplainer
from incx.models.base_model import BaseModel


class ExplainerFactory:
    def __init__(self, model: BaseModel) -> None:
        self._model = model

    def get_explainer(self, explainer: ExplainerEnum) -> BaseExplainer:
        if explainer == ExplainerEnum.D_RISE:
            return DRise(self._model)

        raise ValueError(f"Unsupported explainer identifier: {explainer}")
