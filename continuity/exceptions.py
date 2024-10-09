class ModelNotExists(Exception):
    def __init__(self) -> None:
        super().__init__("Requested Model does not exists.")


class ModelVersionDoesNotExists(Exception):
    def __init__(self) -> None:
        super().__init__("Requested Model Model version does not exists.")

class ModelIntegrityError(Exception):
    def __init__(self) -> None:
        super().__init__("Model checkpoints are not matching.")