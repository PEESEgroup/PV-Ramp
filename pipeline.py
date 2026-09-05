import torch
import torch.nn as nn

from utilities import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one


class PVRampPipeline(nn.Module):
    """Leakage-safe PhyDNet -> diffusion -> RaPVFormer inference pipeline."""

    def __init__(self, phydnet, diffusion, rapvformer):
        super().__init__()
        self.phydnet = phydnet
        self.diffusion = diffusion
        self.rapvformer = rapvformer

    @torch.no_grad()
    def forecast(self, past_rgb, past_sun, future_sun, past_pv, generator=None):
        self.eval()
        coarse = self.phydnet.predict_sequence(past_rgb, past_sun, future_sun)
        past_four = past_rgb[:, -4:].permute(0, 2, 1, 3, 4).contiguous()
        coarse_channel_first = coarse.permute(0, 2, 1, 3, 4).contiguous()
        refined = self.diffusion.sample(
            normalize_to_neg_one_to_one(past_four),
            normalize_to_neg_one_to_one(coarse_channel_first),
            generator=generator,
        )
        refined = unnormalize_to_zero_to_one(refined).clamp(0.0, 1.0)
        refined = refined.permute(0, 2, 1, 3, 4).contiguous()
        future_pv, ramp_logits = self.rapvformer(
            past_rgb, past_sun, past_pv, refined, future_sun
        )
        return refined, future_pv, ramp_logits

    def forward(self, past_rgb, past_sun, future_sun, past_pv, generator=None):
        return self.forecast(past_rgb, past_sun, future_sun, past_pv, generator)
