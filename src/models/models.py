from src.models.baseline import BaselineModel


def get_models():
    return {
        BaselineModel.NAME: BaselineModel,
        'all': None
    }