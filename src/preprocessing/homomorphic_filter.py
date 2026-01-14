import torch


class HomomorphicFilter:
    """
    A PyTorch transform to apply a homomorphic filter to an image.

    Homomorphic filtering is a technique used for image enhancement that corrects
    non-uniform illumination and enhances contrast. It operates in the frequency
    domain to separate illumination (low-frequency) and reflectance (high-frequency)
    components.

    Args:
        a (float): Offset for the filter gain. Controls the contribution of the
                   high-pass filtered component. Default is 0.5.
        b (float): Slope for the filter gain. Controls the sharpness of the
                   transition. Default is 1.5.
        cutoff (int): Cutoff frequency for the high-pass filter.
    """

    def __init__(self, a: float = 0.5, b: float = 1.5, cutoff: int = 32):
        self.a = float(a)
        self.b = float(b)
        self.cutoff = int(cutoff)

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the homomorphic filter.

        Args:
            img_tensor (torch.Tensor): A single-channel image tensor (C, H, W)
                                       with values in the range [0, 1].

        Returns:
            torch.Tensor: The filtered image tensor, normalized to [0, 1].
        """
        if img_tensor.dim() != 3 or img_tensor.shape[0] != 1:
            raise ValueError(
                "HomomorphicFilter expects a single-channel image tensor (C, H, W) with C=1."
            )

        # 1. Log transform the image (add epsilon to avoid log(0))
        img_log = torch.log1p(img_tensor)

        # 2. To frequency domain via Fast Fourier Transform
        img_fft_shifted = torch.fft.fftshift(torch.fft.fft2(img_log))

        # 3. Create a Gaussian high-pass filter
        _, H, W = img_tensor.shape
        u, v = torch.meshgrid(
            torch.arange(0, H, device=img_tensor.device) - H // 2,
            torch.arange(0, W, device=img_tensor.device) - W // 2,
            indexing="ij",
        )
        D_sq = u**2 + v**2
        hpf = 1.0 - torch.exp(-D_sq / (2 * self.cutoff**2))
        hpf_gain = (self.b - self.a) * hpf + self.a

        # 4. Apply filter and inverse FFT
        filtered_fft = torch.fft.ifftshift(img_fft_shifted * hpf_gain)
        filtered_log = torch.real(torch.fft.ifft2(filtered_fft))

        # 5. Reverse log transform and normalize output
        filtered_img = torch.expm1(filtered_log)
        return (filtered_img - filtered_img.min()) / (filtered_img.max() - filtered_img.min() + 1e-8)

