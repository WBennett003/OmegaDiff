import torch


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    a = a.cpu()
    out = a.gather(-1, t.cpu())
    return (out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))).to(t.device)

class Diffusion:
    def __init__(self, timesteps):
        self.timesteps = timesteps

        self.betas = self.cosine_beta_schedule()
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1,0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_varience = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def cosine_beta_schedule(self, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = self.timesteps + 1
        x = torch.linspace(0, self.timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_samples(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, x_t, t, t_index, pred_noise):
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)
        
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_varience, t, pred_noise.shape)
            noise = torch.randn_like(pred_noise)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def masked_q_samples(self, x_0, mask, t, noise=None):
        x_t = self.q_samples(x_0, t, noise)
        x_t = x_0 + x_t * (1 - mask[:, :, None])
        return x_t

def test_masked_noise():
    a = torch.arange(0, 1280).repeat(1280).repeat(2).reshape((2, 1280, 1280))
    mask = torch.ones((2, 1280))
    z = torch.zeros((2, 640))
    mask[:, :639] = z[:, :639]

    print(a.shape, mask.shape)

    d = Diffusion(200)
    noise = torch.randn(a.shape)
    x_t = d.masked_q_samples(a, mask, torch.as_tensor([100, 100]), noise)

    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.heatmap(a[0])
    plt.show()

    sns.heatmap(x_t[0])
    plt.show()

if __name__ == '__main__':
    test_masked_noise()