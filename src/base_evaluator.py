from abc import ABC


class BaseEvaluator(ABC):
    def __init__(self, *args, **kwargs):
        super(BaseEvaluator, self).__init__(*args, **kwargs)

    def evaluate(self, data):
        """
        Abstract method to evaluate the input data.
        """
        raise NotImplementedError

