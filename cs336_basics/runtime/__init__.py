from cs336_basics.runtime.device import default_device

__all__ = ["default_device", "get_batch"]


def __getattr__(name: str):
    if name == "get_batch":
        from cs336_basics.runtime.loader import get_batch

        return get_batch
    raise AttributeError(name)
