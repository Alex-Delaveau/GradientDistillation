import torch


class DeviceSingleton:
    _device = None

    @classmethod
    def get(cls) -> torch.device:
        if cls._device is None:
            cls._device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else
                "cpu"
            )
        return cls._device

    @classmethod
    def device_count(cls) -> int:
        device = cls.get()
        if device.type == "cuda":
            return torch.cuda.device_count()
        return 1

    @classmethod
    def is_distributed(cls) -> bool:
        return torch.cuda.is_available() and torch.cuda.device_count() > 1

    @classmethod
    def get_rng(cls) -> list:
        return torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []

    @classmethod
    def set_rng(cls, state: list) -> None:
        if torch.cuda.is_available() and state:
            torch.cuda.set_rng_state_all(state)

    @classmethod
    def manual_seed(cls, seed: int) -> None:
        torch.manual_seed(seed)
        if cls.get().type == "cuda":
            torch.cuda.manual_seed_all(seed)
