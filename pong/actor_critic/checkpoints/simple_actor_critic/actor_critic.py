import torch.nn as nn


# This actor critic is very simple and something of a placeholder to be replaced later
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Shared encoder
        # The input is expected to be a grayscale 210x160 image (shape [1, 210, 160])
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32x105x80
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64x52x40
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128x26x20
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1, stride=1), # Output: 1x26x20
            nn.Flatten() # Output: 520 (26 * 20 * 1)
        )
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(520, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
            nn.Softmax(dim=-1) # Output is a probability distribution over 6 actions
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(520, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Output is a single value representing the state value
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        actor_output = self.actor(encoded)
        critic_output = self.critic(encoded)
        return actor_output.squeeze(), critic_output.squeeze()