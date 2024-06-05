"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood


def extract(v, t, x_shape=None):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = v.to(t.device)[t].float()
    if x_shape is None:
        return out
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float64)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == 'linear':
        # Linear schedule from ho et al, extended to work for any number of dif
        # fusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=torch.float64).double()
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """
    def __init__(self, *, betas, model_mean_type, model_var_type, loss_type, rescale_timesteps=False):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        betas = betas.double()
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas = alphas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], [1,0], value=1.0)
        self.alphas_cumprod_next = F.pad(self.alphas_cumprod[1:], [0,1], value=0.0)

        # calculations for diffusion q(x_t|x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1}|x_t, x_0)
        self.posterior_variance = betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.cat((self.posterior_variance[1:2], self.posterior_variance[1:])))
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - self.alphas_cumprod)



    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_0: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
                max_log = extract(torch.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else:
            model_variance, model_log_variance =  {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    torch.cat((self.posterior_variance[1:2], self.betas[1:])),
                    torch.log(torch.cat((self.posterior_variance[1:2], self.betas[1:])))
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = extract(model_variance, t, x.shape)
            model_log_variance = extract(model_log_variance, t, x.shape)
        
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(self._predict_xstart_from_xprev(x, t, model_output))
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(self._predict_xstart_from_eps(x, t, model_output))
            model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x, t)
        else:
            raise NotImplementedError(self.model_mean_type)
        
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }


    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps


    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return extract(1. / self.posterior_mean_coef1, t, x_t.shape) * xprev - \
            extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.sahpe) * x_t
    

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    

    def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(model, x, t, clip_denoised, denoised_fn, model_kwargs)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    
    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, 
                      model_kwargs=None, device=None, progress=False):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(model, shape, noise=noise, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs, device=device, progress=progress):
            final = sample
        return final["sample"]


    def p_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True,
             denoised_fn=None, model_kwargs=None, device=None, progress=False):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs)
                yield out
                img = out["sample"]
                #print(i,img.size())

    
    def ddim_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0):
        """
        Sample x_{t-1} from the model using DDIM.
        
        Same usage as p_sample().
        """
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)

        noise = torch.randn_like(x)
        mean_pred = out["pred_xstart"] * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        #sample = mean_pred
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    

    def ddim_reverse_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs)
        # Usually our model outputs epsilon, but we re-derive it 
        # in case we used x_start or x_prev prediction.
        eps = (extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - out["pred_xstart"]) / extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = extract(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = out["pred_xstart"] * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * eps
        
        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}


    def ddim_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(model, shape, noise=noise, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs, device=device, progress=progress, eta=eta):
            final = sample
        return final["sample"]
    

    def ddim_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)


        x_save = []
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs, eta=eta)
                yield out
                img = out["sample"]
    

    def _vb_terms_bpd(self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start, x_t, t)
        out = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = mean_flat(kl) / math.log(2.0)
        decoder_nll = -discretized_gaussian_log_likelihood(x_start, means=out["mean"], log_scales=0.5 * out["log_variance"])
        decoder_nll = mean_flat(decoder_nll) / math.log(2.0) 
        
        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t==0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}


    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(model, x_start, x_t, t, clip_denoised=False, model_kwargs=model_kwargs)["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let 
                # it affect our mean prediction
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0
            
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)
        
        return terms
    

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        "param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / math.log(2.0)
    

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entry variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with torch.no_grad():
                out = self._vb_terms_bpd(model, x_start=x_start, x_t=x_t, t=t_batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))
        
        vb = torch.stack(vb, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


class GaussianMixtureDiffusion(GaussianDiffusion):
    """
    Utilities for training and sampling diffusion models.
    """
    def __init__(self, *, mus, omegas, betas, model_mean_type, model_var_type, loss_type, rescale_timesteps=False):
        super().__init__(model_var_type=model_var_type, betas=betas, model_mean_type=model_mean_type, loss_type=loss_type, rescale_timesteps=rescale_timesteps)
        mus, omegas = torch.tensor(mus), torch.tensor(omegas)
        assert mus.shape[0] == omegas.shape[0]
        self.num_gaussians = mus.shape[0]
        self.mus = mus
        self.omegas = omegas
        self.omegas_cumsum = torch.cumsum(self.omegas, dim=0)

        # calculations for gammas 
        self.gammas = 1.0 - self.alphas_cumprod
        self.gammas_prev = 1.0 - self.alphas_cumprod_prev
        alphas_bar_ti = self.alphas_cumprod.unsqueeze(dim=0).repeat(self.num_timesteps, 1)
        gammas_ti = torch.mul(self.betas.unsqueeze(dim=1).repeat(1, self.num_timesteps),
                             torch.div(alphas_bar_ti, alphas_bar_ti.permute(1, 0))).permute(1, 0)
        self.gammas_ti = torch.tril(gammas_ti)

        # sigmas_2 = betas * gammas_pre / gammas + gammas_pre * (betas / gammas) ** 2 * torch.sum(mus ** 2 * omegas * img_size * img_size * in_channel, dim=0)
        self.sigmas_2 = betas * self.gammas_prev / self.gammas

        # calculations for diffusion q(x_t|x_{t-1}) and others
        self.sqrt_gammas = torch.sqrt(self.gammas)
        self.sqrt_gammas_ti = torch.sqrt(self.gammas_ti)

        # calculations for posterior q(x_{t-1}|x_t,x_0)
        self.posterior_variance_w_mu_diffusion = self.posterior_variance + \
            self.betas ** 2 / self.gammas ** 2 * self.gammas_prev * torch.abs(self.mus).mean()
        self.posterior_log_variance_clipped_w_mu = torch.log(self.posterior_variance_w_mu_diffusion)
        self.posterior_mean_coef3 = self.gammas_prev * torch.sqrt(self.alphas * betas) / self.gammas

        self.sigma_muTmu = (self.gammas[[-1]] * torch.sum(omegas * mus * mus)).item()

        self.classifier_lf = nn.NLLLoss()
        self.logsoft = nn.LogSoftmax(dim=1)


    def get_mu_diffusion(self, x_start, t):
        N = x_start.shape[0]
        T = self.num_timesteps
        K = self.num_gaussians
        device = t.device
        size = (torch.tensor(x_start.shape[1:]).cumprod(dim=0))[-1].float().sqrt().to(device)

        probs = 1 - torch.rand(size=(N, T), device=device) # N x T
        mask = torch.arange(T, device=device).view(1,-1).repeat(N,1) <= t.view(-1,1)
        probs = probs.mul(mask).unsqueeze(-1).repeat(1, 1, K) # N x T x K

        omegas_up = self.omegas_cumsum.view(1, 1, K).repeat(N, T, 1).to(device)
        omegas_low = (self.omegas_cumsum - self.omegas).view(1, 1, K).repeat(N, T, 1).to(device)
        mask = (probs > omegas_low) & (probs <= omegas_up)

        mus = self.mus.view(1, 1, K).repeat(N, T, 1).to(device) / size
        mus_selected = mus.mul(mask).sum(dim=-1) # N x T

        mask2_sup1 = torch.sum((torch.arange(K) + 1).view(1, 1, K).repeat(N, T, 1).to(device).mul(mask),dim=2)
        logis = torch.sum(torch.zeros(N, T).to(device).scatter_(1, t.unsqueeze(1), 1).mul(mask2_sup1),dim=1)-1

        sqrt_gammas_ti = extract(self.sqrt_gammas_ti, t)
        diffusion = mus_selected.mul(sqrt_gammas_ti).sum(dim=-1).view(N, 1, 1, 1)
        #diffusion = diffusion.view(N, [1] * (len(x_start.shape) - 1))
        return diffusion, logis.to(torch.int64)
    

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        mu_diff, _ = self.get_mu_diffusion(x_start, t)
        x_t = self.q_sample(x_start, t, noise=noise) + mu_diff
        terms = {}
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(model, x_start, x_t, t, clip_denoised=False, model_kwargs=model_kwargs)["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let 
                # it affect our mean prediction
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0
            
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)
        
        return terms


    def training_classifier(self, classifier_model, x_start, t, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        mu_diff, logis = self.get_mu_diffusion(x_start, t)
        x_t = self.q_sample(x_start, t, noise=noise) + mu_diff

        terms = {}

        classifier_res = classifier_model(x_t, t)
        classifier_loss = self.classifier_lf(self.logsoft(classifier_res), logis)
        terms["loss"] = classifier_loss

        return terms


    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None,
                      model_kwargs=None, device=None, progress=False, sample_mode='removal_mu', classifier_model=None):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param sample_mode: original, removal_mu, standard
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(model, shape, noise=noise, clip_denoised=clip_denoised,
                                                     denoised_fn=denoised_fn, model_kwargs=model_kwargs, device=device,
                                                     progress=progress, sample_mode=sample_mode, classifier_model=classifier_model):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True,
                                  denoised_fn=None, model_kwargs=None, device=None, progress=False, sample_mode='removal_mu', classifier_model=None):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        if noise is not None:
            img = noise
        elif sample_mode == 'original':
            img = torch.randn(*shape, device=device)
        else:
            img = torch.randn(*shape, device=device)
            size = (torch.tensor(img.shape[1:]).cumprod(dim=0))[-1].float().sqrt().to(device)
            img = img + (self.sigma_muTmu * torch.rand(1)).to(device)/size
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        if sample_mode == 'standard':
            for i in indices:
                t = torch.tensor([i] * shape[0], device=device)
                with torch.no_grad():
                    out = self.p_sample_standard(model, classifier_model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                                    model_kwargs=model_kwargs)
                    yield out
                    img = out["sample"]
        else:
            for i in indices:
                t = torch.tensor([i] * shape[0], device=device)
                with torch.no_grad():
                    out = self.p_sample(model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                                    model_kwargs=model_kwargs)
                    yield out
                    img = out["sample"]
        
    def p_sample_standard(self, model, classifier_model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(model, x, t, clip_denoised, denoised_fn, model_kwargs)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        mus_selected = self.get_mu_denoising(x, t)
        sample = out["mean"] - mus_selected + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def get_mu_denoising(self, x, t):
        N = x.shape[0]
        K = self.num_gaussians
        device = t.device
        size = (torch.tensor(x.shape[1:]).cumprod(dim=0))[-1].float().sqrt().to(device)

        beta_t = self.betas[t].to(device)
        alpha_t = self.alphas[t].to(device)
        probs = 1 - torch.rand(size=(N, 1), device=device)
        probs = probs.repeat(1, K)

        omegas_up = self.omegas_cumsum.view(1, K).repeat(N, 1).to(device)
        omegas_low = (self.omegas_cumsum - self.omegas).view(1, K).repeat(N, 1).to(device)
        mask = (probs > omegas_low) & (probs <= omegas_up)

        mus = self.mus.view(1, K).repeat(N, 1).to(device) / size
        mus_selected = mus.mul(mask).sum(dim=-1)
        out = torch.sqrt(beta_t/alpha_t) * mus_selected
        out = out.to(mus_selected.dtype)
        return out.view(N, 1, 1, 1)