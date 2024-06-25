from incremental_explainer.explainers.explainer_enum import ExplainerEnum
from incremental_explainer.explainers.d_rise import DRise
from incremental_explainer.explainers.base_explainer import BaseExplainer


class ExplainerFactory:
    
    def __init__(self, results) -> None:
        self._results = results
    
    def get_explainer(self, explainer: ExplainerEnum) -> BaseExplainer:
        if explainer == ExplainerEnum.D_RISE:
            return DRise(self._results)

        raise ValueError(f"Unsupported explainer identifier: {explainer}")