from abc import ABC, abstractmethod
from torch import Tensor


class JParameterization(ABC):
    """Parameterization strategy for the scene radiance J.

    The host (PhysicsFormationDataset) never sees the internal structure:
    it only decodes, collects parameters, and asks for extension.
    """

    @abstractmethod
    def decode(self, n_levels: int | None = None) -> Tensor:
        """Render J in [0,1], shape [N, 3, syn_res, syn_res]."""
    
    @abstractmethod
    def parameters(self) -> list[Tensor]:
        """Trainable leaf tensors, for the optimizer."""


    def extend(self) -> bool:
        """Progressive-growth hook. Returns True if parameters changed
        (host must rebuild its optimizer). Default: nothing to do."""
        return False

    @abstractmethod
    def state_dict(self) -> dict: ...

    @abstractmethod
    def load_state_dict(self, d: dict) -> None: ...

    def snapshot_levels(self) -> tuple[list[Tensor], list[int]]:
        """Decoded J per cumulative level, and their resolutions.
        Default: single-level parameterizations return one entry.
        Multi-resolution strategies (pyramid) override this."""
        J = self.decode()
        return [J], [J.shape[-1]]