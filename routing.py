from typing import Final

import torch


class MMRouting(torch.nn.Module):
    def __init__(
        self,
        *,
        in_capsules: int = -1,
        input_size: int = -1,
        out_capsules: int = -1,
        output_size: int = -1,
        iterations: int = 10,
    ) -> None:
        """
        Args:
            in_capsuls:
                Number of modality combinations (7 for three modalities).
            input_size:
                Input embedding dimension for each capsule.
            out_capsules:
                Number of concepts/classes.
            output_size:
                Output embedding dimension for each capsule.
        """
        super().__init__()
        self.in_capsules: Final = in_capsules
        self.input_size: Final = input_size
        self.out_capsules: Final = out_capsules
        self.output_size: Final = output_size
        self.iterations: Final = iterations

        self.weights = torch.nn.Parameter(
            torch.randn(in_capsules, input_size, out_capsules, output_size)
        )

    def forward(
        self,
        *,
        f_bid: torch.Tensor,
        p_bi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # b:batch, i: modalities, j: concepts, d: modality features, e: concept features
        # initialize concepts
        c_bj = torch.ones(1, device=f_bid.device, dtype=f_bid.dtype).expand(
            f_bid.shape[0], self.out_capsules, self.output_size
        )
        r_bij = c_bj  # for jit

        f_W_bije = torch.einsum("bid, idje -> bije", f_bid, self.weights)
        for _ in range(self.iterations):
            r_bij = torch.nn.functional.softmax(
                torch.einsum("bije, bje -> bij", f_W_bije, c_bj), dim=-1
            )
            c_bj = torch.einsum("bi, bij, bije -> bje", p_bi, r_bij, f_W_bije)
        return c_bj, r_bij
