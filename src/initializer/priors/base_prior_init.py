from abc import ABC, abstractmethod
from torch import Tensor


class BasePriorInitializer(ABC):

    @abstractmethod
    def load_img(self, path: str) -> Tensor:
        """path -> tensor au format attendu par get_priors (spécifique au modèle)."""
        ...

    @abstractmethod
    def get_priors(self, image) -> dict:
        """image -> {"T":..., "B":...} (+ "J" optionnel) en [0,1], résolution native."""
        ...

    @staticmethod
    @abstractmethod
    def compose(J: Tensor, T: Tensor, B: Tensor) -> Tensor:
        """Recombine les priors selon le modèle de formation de CET initializer."""
        ...


    def get_priors_from_path(self, path: str) -> dict:
        return self.get_priors(self.load_img(path))

    def get_priors_batch(self, images: list) -> list:
        return [self.get_priors(img) for img in images]