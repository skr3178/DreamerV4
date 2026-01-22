"""
Agent Heads for DreamerV4 Phase 2

MLP heads attached to the frozen transformer for:
- Policy (action prediction)
- Value (state value estimation)
- Reward (reward prediction)

These heads are trained during Phase 2 (Agent Finetuning) and Phase 3 (Imagination RL)
while the main transformer remains frozen.
"""

from typing import Optional, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class PolicyHead(nn.Module):
    """
    Policy head for action prediction.
    
    Supports both:
    - Standard mode: Predicts action for each timestep independently
    - MTP mode: Predicts L+1 future actions from a single embedding h_t (Multi-Token Prediction)
    
    Outputs action distribution conditioned on latent state.
    Supports:
    - Discrete: Single categorical distribution
    - Multi-discrete: Binary keyboard (8 independent Bernoulli) + Categorical camera (121 classes with foveated discretization)
    - Continuous: Normal distribution
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_discrete_actions: Optional[int] = None,
        continuous_action_dim: Optional[int] = None,
        use_multi_discrete: bool = False,  # If True: keyboard (8 binary) + camera (121 categorical)
        num_layers: int = 2,
        mtp_length: int = 8,
        use_mtp: bool = True,
    ):
        """
        Args:
            input_dim: Dimension of input latent representation
            hidden_dim: Hidden layer dimension
            num_discrete_actions: Number of discrete actions (if discrete, single categorical)
            continuous_action_dim: Dimension of continuous actions (if continuous)
            use_multi_discrete: If True, use multi-discrete format (8 binary keyboard + 121 categorical camera)
            num_layers: Number of hidden layers
            mtp_length: Length L for MTP (predicts n=0 to L, so L+1 timesteps per Equation 9)
            use_mtp: Whether to use Multi-Token Prediction mode (default True per paper)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_discrete_actions = num_discrete_actions
        self.continuous_action_dim = continuous_action_dim
        self.use_multi_discrete = use_multi_discrete
        self.mtp_length = mtp_length
        self.use_mtp = use_mtp
        
        # Build MLP layers
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Output heads
        if use_multi_discrete:
            # Multi-discrete: 8 binary keyboard + 121 categorical camera
            num_keyboard = 8
            num_camera = 121  # 11×11 foveated discretization
            
            if use_mtp:
                # MTP: predict L+1 timesteps from single h_t
                self.keyboard_head = nn.Linear(hidden_dim, (mtp_length + 1) * num_keyboard)
                self.camera_head = nn.Linear(hidden_dim, (mtp_length + 1) * num_camera)
            else:
                # Standard: predict single timestep
                self.keyboard_head = nn.Linear(hidden_dim, num_keyboard)
                self.camera_head = nn.Linear(hidden_dim, num_camera)
        
        elif num_discrete_actions is not None:
            if use_mtp:
                # MTP: predict L+1 timesteps from single h_t
                self.action_head = nn.Linear(hidden_dim, (mtp_length + 1) * num_discrete_actions)
            else:
                # Standard: predict single timestep
                self.action_head = nn.Linear(hidden_dim, num_discrete_actions)
        
        if continuous_action_dim is not None:
            if use_mtp:
                # MTP: predict L+1 timesteps
                self.mean_head = nn.Linear(hidden_dim, (mtp_length + 1) * continuous_action_dim)
                self.log_std_head = nn.Linear(hidden_dim, (mtp_length + 1) * continuous_action_dim)
            else:
                # Standard: predict single timestep
                self.mean_head = nn.Linear(hidden_dim, continuous_action_dim)
                self.log_std_head = nn.Linear(hidden_dim, continuous_action_dim)
    
    def forward(
        self,
        latents: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute action distribution from latent state.
        
        Args:
            latents: Latent representation 
                - Standard mode: (batch, ..., input_dim)
                - MTP mode: (batch, input_dim) - single h_t embedding
        
        Returns:
            Dictionary containing distribution parameters:
                - Multi-discrete mode:
                    - keyboard_logits: (batch, ..., 8) or (batch, mtp_length+1, 8)
                    - camera_logits: (batch, ..., 121) or (batch, mtp_length+1, 121)
                - Discrete mode:
                    - logits: (batch, ..., num_actions) or (batch, mtp_length+1, num_actions)
                    - probs: (batch, ..., num_actions) or (batch, mtp_length+1, num_actions)
                - Continuous mode:
                    - mean, std, log_std: (batch, ..., action_dim) or (batch, mtp_length+1, action_dim)
        """
        # Flatten if needed
        original_shape = latents.shape[:-1]
        latents = latents.reshape(-1, self.input_dim)
        
        # MLP forward
        hidden = self.mlp(latents)
        
        result = {}
        
        if self.use_multi_discrete:
            # Multi-discrete: keyboard (binary) + camera (categorical)
            keyboard_logits = self.keyboard_head(hidden)
            camera_logits = self.camera_head(hidden)
            
            if self.use_mtp:
                # MTP: reshape to (batch, mtp_length+1, ...)
                batch_size = keyboard_logits.shape[0]
                keyboard_logits = keyboard_logits.reshape(batch_size, self.mtp_length + 1, 8)
                camera_logits = camera_logits.reshape(batch_size, self.mtp_length + 1, 121)
            else:
                # Standard: reshape to original shape
                keyboard_logits = keyboard_logits.reshape(*original_shape, 8)
                camera_logits = camera_logits.reshape(*original_shape, 121)
            
            result["keyboard_logits"] = keyboard_logits
            result["camera_logits"] = camera_logits
            result["camera_probs"] = F.softmax(camera_logits, dim=-1)
        
        elif self.num_discrete_actions is not None:
            logits = self.action_head(hidden)
            
            if self.use_mtp:
                # MTP: reshape to (batch, mtp_length+1, num_actions)
                batch_size = logits.shape[0]
                logits = logits.reshape(batch_size, self.mtp_length + 1, self.num_discrete_actions)
            else:
                # Standard: reshape to original shape
                logits = logits.reshape(*original_shape, self.num_discrete_actions)
            
            result["logits"] = logits
            result["probs"] = F.softmax(logits, dim=-1)
        
        if self.continuous_action_dim is not None:
            mean = self.mean_head(hidden)
            log_std = self.log_std_head(hidden)
            log_std = torch.clamp(log_std, -20, 2)  # Stability
            
            if self.use_mtp:
                # MTP: reshape to (batch, mtp_length+1, action_dim)
                batch_size = mean.shape[0]
                mean = mean.reshape(batch_size, self.mtp_length + 1, self.continuous_action_dim)
                log_std = log_std.reshape(batch_size, self.mtp_length + 1, self.continuous_action_dim)
            else:
                # Standard: reshape to original shape
                mean = mean.reshape(*original_shape, self.continuous_action_dim)
                log_std = log_std.reshape(*original_shape, self.continuous_action_dim)
            
            result["mean"] = mean
            result["log_std"] = log_std
            result["std"] = torch.exp(log_std)
        
        return result
    
    def sample(
        self,
        latents: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        """
        Sample actions from the policy.
        
        Args:
            latents: Latent representation
            deterministic: If True, return mode of distribution
        
        Returns:
            actions: Sampled actions
                - Multi-discrete: Dict with 'keyboard' (batch, ..., 8) and 'camera' (batch, ...)
                - Discrete: (batch, ...)
                - Continuous: (batch, ..., action_dim)
            log_probs: Log probabilities of sampled actions
        """
        dist_params = self.forward(latents)
        
        if self.use_multi_discrete:
            from torch.distributions import Independent, Bernoulli, Categorical
            
            keyboard_logits = dist_params["keyboard_logits"]
            camera_logits = dist_params["camera_logits"]
            
            # Sample keyboard (8 independent Bernoulli)
            keyboard_dist = Independent(Bernoulli(logits=keyboard_logits), 1)
            if deterministic:
                keyboard_actions = (keyboard_logits > 0).long()
            else:
                keyboard_actions = keyboard_dist.sample().long()
            keyboard_log_probs = keyboard_dist.log_prob(keyboard_actions.float())
            
            # Sample camera (categorical)
            camera_dist = Categorical(logits=camera_logits)
            if deterministic:
                camera_actions = camera_logits.argmax(dim=-1)
            else:
                camera_actions = camera_dist.sample()
            camera_log_probs = camera_dist.log_prob(camera_actions)
            
            actions = {
                "keyboard": keyboard_actions,
                "camera": camera_actions,
            }
            log_probs = keyboard_log_probs + camera_log_probs
            
        elif self.num_discrete_actions is not None:
            logits = dist_params["logits"]
            dist = Categorical(logits=logits)
            
            if deterministic:
                actions = logits.argmax(dim=-1)
            else:
                actions = dist.sample()
            
            log_probs = dist.log_prob(actions)
            
        else:  # Continuous
            mean = dist_params["mean"]
            std = dist_params["std"]
            dist = Normal(mean, std)
            
            if deterministic:
                actions = mean
            else:
                actions = dist.rsample()  # Reparameterized sample
            
            log_probs = dist.log_prob(actions).sum(dim=-1)
        
        return actions, log_probs
    
    def log_prob(
        self,
        latents: torch.Tensor,
        actions: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute log probability of actions under the policy.
        
        Args:
            latents: Latent representation
            actions: Actions to evaluate
                - Multi-discrete: Dict with 'keyboard' and 'camera'
                - Discrete: (batch, ...)
                - Continuous: (batch, ..., action_dim)
        
        Returns:
            log_probs: Log probabilities
        """
        dist_params = self.forward(latents)
        
        if self.use_multi_discrete:
            from torch.distributions import Independent, Bernoulli, Categorical
            
            keyboard_logits = dist_params["keyboard_logits"]
            camera_logits = dist_params["camera_logits"]
            
            # Keyboard log prob
            keyboard_dist = Independent(Bernoulli(logits=keyboard_logits), 1)
            keyboard_log_probs = keyboard_dist.log_prob(actions["keyboard"].float())
            
            # Camera log prob
            camera_dist = Categorical(logits=camera_logits)
            camera_log_probs = camera_dist.log_prob(actions["camera"])
            
            return keyboard_log_probs + camera_log_probs
        
        elif self.num_discrete_actions is not None:
            dist = Categorical(logits=dist_params["logits"])
            return dist.log_prob(actions)
        else:
            dist = Normal(dist_params["mean"], dist_params["std"])
            return dist.log_prob(actions).sum(dim=-1)


class ValueHead(nn.Module):
    """
    Value head for state value estimation V(s).
    
    Outputs a scalar value estimate for each state.
    Uses distributional value learning with symlog encoding.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_bins: int = 255,
        value_range: Tuple[float, float] = (-20.0, 20.0),
        use_symlog: bool = True,
    ):
        """
        Args:
            input_dim: Dimension of input latent representation
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            num_bins: Number of bins for distributional value
            value_range: (min, max) value range
            use_symlog: Whether to use symlog bin spacing (default True per paper Appendix)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_bins = num_bins
        self.value_range = value_range
        self.use_symlog = use_symlog
        
        # Build MLP layers
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Output head (distributional)
        self.value_head = nn.Linear(hidden_dim, num_bins)
        
        # Precompute bin centers with symlog spacing
        if use_symlog:
            # Create bins in symlog space, then convert back
            symlog_min = self._symlog(torch.tensor(value_range[0]))
            symlog_max = self._symlog(torch.tensor(value_range[1]))
            symlog_bin_centers = torch.linspace(symlog_min.item(), symlog_max.item(), num_bins)
            bin_centers = self._symexp(symlog_bin_centers)
        else:
            # Linear spacing (backward compatibility)
            bin_centers = torch.linspace(value_range[0], value_range[1], num_bins)
        self.register_buffer("bin_centers", bin_centers)
    
    @staticmethod
    def _symlog(x: torch.Tensor) -> torch.Tensor:
        """Symmetric logarithm: sign(x) * ln(|x| + 1)"""
        return torch.sign(x) * torch.log1p(torch.abs(x))
    
    @staticmethod
    def _symexp(x: torch.Tensor) -> torch.Tensor:
        """Inverse of symlog: sign(x) * (exp(|x|) - 1)"""
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    
    def forward(self, latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute value distribution and expected value.
        
        Args:
            latents: Latent representation (batch, ..., input_dim)
        
        Returns:
            Dictionary containing:
                - logits: Distribution logits
                - probs: Distribution probabilities
                - value: Expected value
        """
        original_shape = latents.shape[:-1]
        latents = latents.reshape(-1, self.input_dim)
        
        hidden = self.mlp(latents)
        logits = self.value_head(hidden)
        
        probs = F.softmax(logits, dim=-1)
        value = (probs * self.bin_centers).sum(dim=-1)
        
        # Reshape outputs
        logits = logits.reshape(*original_shape, self.num_bins)
        probs = probs.reshape(*original_shape, self.num_bins)
        value = value.reshape(*original_shape)
        
        return {
            "logits": logits,
            "probs": probs,
            "value": value,
        }
    
    def target_to_bins(self, target: torch.Tensor) -> torch.Tensor:
        """
        Convert target values to bin distribution targets (two-hot encoding).
        
        Args:
            target: Target values (batch, ...)
        
        Returns:
            Bin distribution targets for cross-entropy loss
        """
        # Clamp target to valid range
        target = target.clamp(self.value_range[0], self.value_range[1])
        
        if self.use_symlog:
            # Convert to symlog space for binning
            symlog_target = self._symlog(target)
            symlog_min = self._symlog(torch.tensor(self.value_range[0], device=target.device))
            symlog_max = self._symlog(torch.tensor(self.value_range[1], device=target.device))

            # Find bin indices in symlog space
            bin_width = (symlog_max - symlog_min) / (self.num_bins - 1)
            bin_idx = (symlog_target - symlog_min) / bin_width
        else:
            # Linear binning (backward compatibility)
            bin_width = (self.value_range[1] - self.value_range[0]) / (self.num_bins - 1)
            bin_idx = (target - self.value_range[0]) / bin_width

        # Two-hot encoding
        lower_idx = bin_idx.floor().long().clamp(0, self.num_bins - 2)
        upper_idx = lower_idx + 1
        upper_weight = bin_idx - lower_idx.float()
        lower_weight = 1.0 - upper_weight

        # Create target distribution
        target_dist = torch.zeros(*target.shape, self.num_bins, device=target.device)
        target_dist.scatter_(-1, lower_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
        target_dist.scatter_(-1, upper_idx.unsqueeze(-1), upper_weight.unsqueeze(-1))

        return target_dist


class RewardHead(nn.Module):
    """
    Reward head for reward prediction r(s).
    
    Supports both:
    - Standard mode: Predicts reward for each timestep independently
    - MTP mode: Predicts L+1 future rewards from a single embedding h_t (Multi-Token Prediction)
    
    Predicts expected reward at each state.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_bins: int = 255,
        reward_range: Tuple[float, float] = (-10.0, 10.0),
        mtp_length: int = 8,
        use_mtp: bool = True,
        use_symlog: bool = True,
    ):
        """
        Args:
            input_dim: Dimension of input latent representation
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            num_bins: Number of bins for distributional reward
            reward_range: (min, max) reward range
            mtp_length: Length L for MTP (predicts n=0 to L, so L+1 timesteps per Equation 9)
            use_mtp: Whether to use Multi-Token Prediction mode (default True per paper)
            use_symlog: Whether to use symlog bin spacing (default True per paper Appendix)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_bins = num_bins
        self.reward_range = reward_range
        self.mtp_length = mtp_length
        self.use_mtp = use_mtp
        self.use_symlog = use_symlog
        
        # Build MLP layers
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Output head
        if use_mtp:
            # MTP: predict L+1 timesteps from single h_t
            self.reward_head = nn.Linear(hidden_dim, (mtp_length + 1) * num_bins)
        else:
            # Standard: predict single timestep
            self.reward_head = nn.Linear(hidden_dim, num_bins)
        
        # Precompute bin centers with symlog spacing
        if use_symlog:
            # Create bins in symlog space, then convert back
            symlog_min = self._symlog(torch.tensor(reward_range[0]))
            symlog_max = self._symlog(torch.tensor(reward_range[1]))
            symlog_bin_centers = torch.linspace(symlog_min.item(), symlog_max.item(), num_bins)
            bin_centers = self._symexp(symlog_bin_centers)
        else:
            # Linear spacing (backward compatibility)
            bin_centers = torch.linspace(reward_range[0], reward_range[1], num_bins)
        self.register_buffer("bin_centers", bin_centers)

    @staticmethod
    def _symlog(x: torch.Tensor) -> torch.Tensor:
        """Symmetric logarithm: sign(x) * ln(|x| + 1)"""
        return torch.sign(x) * torch.log1p(torch.abs(x))
    
    @staticmethod
    def _symexp(x: torch.Tensor) -> torch.Tensor:
        """Inverse of symlog: sign(x) * (exp(|x|) - 1)"""
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    
    def forward(self, latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute reward distribution and expected reward.
        
        Args:
            latents: Latent representation
                - Standard mode: (batch, ..., input_dim)
                - MTP mode: (batch, input_dim) - single h_t embedding
        
        Returns:
            Dictionary containing:
                - logits: Distribution logits
                    - Standard mode: (batch, ..., num_bins)
                    - MTP mode: (batch, mtp_length+1, num_bins) - predictions for n=0 to L
                - probs: Distribution probabilities
                - reward: Expected reward
        """
        original_shape = latents.shape[:-1]
        latents = latents.reshape(-1, self.input_dim)
        
        hidden = self.mlp(latents)
        logits = self.reward_head(hidden)
        
        if self.use_mtp:
            # MTP: reshape to (batch, mtp_length+1, num_bins)
            batch_size = logits.shape[0]
            logits = logits.reshape(batch_size, self.mtp_length + 1, self.num_bins)
        else:
            # Standard: reshape to original shape
            logits = logits.reshape(*original_shape, self.num_bins)
        
        probs = F.softmax(logits, dim=-1)
        reward = (probs * self.bin_centers).sum(dim=-1)
        
        return {
            "logits": logits,
            "probs": probs,
            "reward": reward,
        }
    
    def target_to_bins(self, target: torch.Tensor) -> torch.Tensor:
        """Convert target rewards to bin distribution targets."""
        target = target.clamp(self.reward_range[0], self.reward_range[1])

        if self.use_symlog:
            # Convert to symlog space for binning
            symlog_target = self._symlog(target)
            symlog_min = self._symlog(torch.tensor(self.reward_range[0], device=target.device))
            symlog_max = self._symlog(torch.tensor(self.reward_range[1], device=target.device))

            # Find bin indices in symlog space
            bin_width = (symlog_max - symlog_min) / (self.num_bins - 1)
            bin_idx = (symlog_target - symlog_min) / bin_width
        else:
            # Linear binning (backward compatibility)
            bin_width = (self.reward_range[1] - self.reward_range[0]) / (self.num_bins - 1)
            bin_idx = (target - self.reward_range[0]) / bin_width

        lower_idx = bin_idx.floor().long().clamp(0, self.num_bins - 2)
        upper_idx = lower_idx + 1
        upper_weight = bin_idx - lower_idx.float()
        lower_weight = 1.0 - upper_weight

        target_dist = torch.zeros(*target.shape, self.num_bins, device=target.device)
        target_dist.scatter_(-1, lower_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
        target_dist.scatter_(-1, upper_idx.unsqueeze(-1), upper_weight.unsqueeze(-1))

        return target_dist
