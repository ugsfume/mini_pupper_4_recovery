"""PPO with a critic warm-up, a bounded adaptive learning rate and smoothness losses on the actor's mean.

The update is RSL-RL's (``rsl_rl.algorithms.ppo.PPO``, rsl-rl-lib 5.0.1) with three additions:

- ``actor_warmup_iters``: the first updates train the value function only. A fine-tune starts with a critic that does
  not know the new rewards; its first policy gradients would otherwise be noise.
- ``lr_min`` / ``lr_max``: the bounds of the KL-adaptive learning rate. Fine-tunes hold it at 5e-5 or below.
- ``caps_temporal`` and ``caps_spatial``: the CAPS and L2C2 smoothness losses. The temporal term is
  mean ||mu(o_t+1) - mu(o_t)||^2 over consecutive rollout steps of one env (not across a reset); the spatial term is
  mean ||mu(o) - mu(o + eps)||^2 with eps ~ N(0, ``caps_obs_noise``^2) in the actor's normalised input units.

With all of them at their defaults the update is the stock one.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms.ppo import PPO


def caps_pair_indices(dones: torch.Tensor) -> torch.Tensor:
    """The flat indices t * N + e (into the storage flattened over (T, N)) of the pairs (t, t + 1) of one env with the
    episode not ended at t: ``dones`` (T, N[, 1]), dones[t, e] the flag stored with step t's transition (the episode
    ended after step t's action, so o_{t+1} is a reset observation). t runs to T - 2; the partner is index + N."""
    d = dones.reshape(dones.shape[0], -1)
    valid = d[:-1] == 0
    return torch.nonzero(valid.reshape(-1), as_tuple=False).reshape(-1)


def actor_mean_from_latent(actor, latent: torch.Tensor) -> torch.Tensor:
    """The actor's deterministic (mean) output from its normalised input (rsl_rl ``MLPModel``: ``get_latent`` then
    ``mlp``, then the distribution's ``deterministic_output``)."""
    out = actor.mlp(latent)
    return actor.distribution.deterministic_output(out) if actor.distribution is not None else out


def caps_temporal_loss(actor, flat_obs, idx: torch.Tensor, num_envs: int) -> torch.Tensor:
    """mean over the pairs ``idx`` (``caps_pair_indices``) of ||mu(o[idx + num_envs]) - mu(o[idx])||^2, the squared
    norm summed over the actions; ``flat_obs`` the storage's observations flattened over (T, N)."""
    lat = actor.get_latent(flat_obs[torch.cat([idx, idx + num_envs])])
    mu0, mu1 = actor_mean_from_latent(actor, lat).chunk(2)
    return (mu1 - mu0).pow(2).sum(dim=-1).mean()


def caps_spatial_loss(actor, obs, clean_mu: torch.Tensor, sigma: float,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """mean over the samples of ||clean_mu - mu(z + eps)||^2 (summed over the actions), z = the actor's normalised input
    of ``obs``, eps ~ N(0, sigma^2) per input float."""
    lat = actor.get_latent(obs)
    eps = torch.randn(lat.shape, generator=generator, device=lat.device, dtype=lat.dtype)
    return (clean_mu - actor_mean_from_latent(actor, lat + sigma * eps)).pow(2).sum(dim=-1).mean()


class SmoothPPO(PPO):
    def __init__(self, actor, critic, storage, *, actor_warmup_iters: int = 0, caps_temporal: float = 0.0,
                 caps_spatial: float = 0.0, caps_obs_noise: float = 0.0, caps_seed: int = 0, lr_min: float = 1.0e-5,
                 lr_max: float = 1.0e-2, **kwargs) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        if self.rnd is not None or self.symmetry is not None or actor.is_recurrent or critic.is_recurrent:
            raise ValueError("SmoothPPO supports plain MLP actors and critics only")
        if min(caps_temporal, caps_spatial, caps_obs_noise) < 0.0:
            raise ValueError("caps_temporal, caps_spatial and caps_obs_noise must be >= 0")
        if caps_spatial > 0.0 and caps_obs_noise <= 0.0:
            raise ValueError("caps_spatial > 0 needs caps_obs_noise > 0")
        if not 0.0 < lr_min <= lr_max:
            raise ValueError(f"need 0 < lr_min <= lr_max (got {lr_min}, {lr_max})")
        self._update_count = 0
        self.actor_warmup_iters = int(actor_warmup_iters)
        self.caps_temporal = float(caps_temporal)
        self.caps_spatial = float(caps_spatial)
        self.caps_obs_noise = float(caps_obs_noise)
        self.lr_min = float(lr_min)
        self.lr_max = float(lr_max)
        self._caps_gen = None
        if self.caps_temporal > 0.0 or self.caps_spatial > 0.0:
            self._caps_gen = torch.Generator(device=self.device)
            self._caps_gen.manual_seed(int(caps_seed))

    def update(self) -> dict[str, float]:
        """rsl_rl's PPO update with the actor warm-up, the smoothness terms and the learning-rate bounds added."""
        warmup = self._update_count < self.actor_warmup_iters
        self._update_count += 1

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_actor_grad_norm = 0.0
        mean_caps_temporal = 0.0
        mean_caps_spatial = 0.0
        mean_kl = 0.0
        mean_lr = 0.0

        do_temporal = not warmup and self.caps_temporal > 0.0
        do_spatial = not warmup and self.caps_spatial > 0.0
        if do_temporal:
            caps_pairs = caps_pair_indices(self.storage.dones)
            caps_obs = self.storage.observations.select(*self.actor.obs_groups).flatten(0, 1)
            do_temporal = caps_pairs.numel() > 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for k, batch in enumerate(generator):
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive" and not warmup:
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(self.lr_min, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(self.lr_max, self.learning_rate * 1.5)
                        # also clamps a start rate outside the bounds
                        self.learning_rate = min(self.lr_max, max(self.lr_min, self.learning_rate))
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
                mean_kl += float(kl_mean)
            else:
                with torch.inference_mode():
                    mean_kl += float(torch.mean(self.actor.get_kl_divergence(batch.old_distribution_params,
                                                                              distribution_params)))
            mean_lr += float(self.optimizer.param_groups[0]["lr"])

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            if warmup:          # value function only: the actor receives no gradient
                loss = self.value_loss_coef * value_loss

            if do_temporal:
                if k % self.num_mini_batches == 0:      # each epoch: every valid pair once, split as the mini-batches
                    perm = torch.randperm(caps_pairs.numel(), generator=self._caps_gen, device=caps_pairs.device)
                    caps_chunks = caps_pairs[perm].tensor_split(self.num_mini_batches)
                idx = caps_chunks[k % self.num_mini_batches]
                if idx.numel() > 0:
                    t_term = caps_temporal_loss(self.actor, caps_obs, idx, self.storage.num_envs)
                    loss = loss + self.caps_temporal * t_term
                    mean_caps_temporal += t_term.item()
            if do_spatial:
                clean_mu = self.actor.output_mean[:original_batch_size]
                obs_mb = batch.observations[:original_batch_size]
                s_term = caps_spatial_loss(self.actor, obs_mb, clean_mu, self.caps_obs_noise, self._caps_gen)
                loss = loss + self.caps_spatial * s_term
                mean_caps_spatial += s_term.item()
            self.optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_actor_grad_norm += float(actor_grad_norm)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_actor_grad_norm /= num_updates
        mean_caps_temporal /= num_updates
        mean_caps_spatial /= num_updates
        mean_kl /= num_updates
        mean_lr /= num_updates

        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "actor_grad_norm": mean_actor_grad_norm,
            "kl": mean_kl,
            "lr_mean": mean_lr,
        }
        if self._caps_gen is not None:      # before lambda; 0 in the warm-up updates
            loss_dict["caps_temporal"] = mean_caps_temporal
            loss_dict["caps_spatial"] = mean_caps_spatial
        return loss_dict
