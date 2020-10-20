import torch

class Aggregate:

    def __init__(self, mode: str='mean') -> None:
        self.mode = mode

    def mean(self, outputs: torch.Tensor) -> torch.Tensor:
        return outputs.mean(axis=1)

    def last(self, outputs: torch.Tensor) -> torch.Tensor:
        return outputs[:,-1,:]

    def __call__(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean":
            return self.mean(outputs)
        elif self.mode == "last":
            return self.last(outputs)
