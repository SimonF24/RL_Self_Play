from collections import deque
from pathlib import Path
from pettingzoo.atari import pong_v3
import random
import supersuit
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from actor_critic import ActorCritic


CHECKPOINT_TO_LOAD: str | None = None
# Relative path to a model checkpoint to load before training (e.g. "checkpoints/actor_critic_episode_10.pth").
# Set to None to train from scratch.
ENTROPY_COEFFICIENT = 0.01 # Coefficient for entropy regularization to encourage exploration
GAMMA = 0.99 # Discount factor for future rewards/state values
GAE_LAMBDA = 0.95 # Lambda parameter for GAE
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 0.5 # Max gradient norm for clipping to improve training stability
NUM_EPISODES = 2000
# Each episode is one game of first to 21 points
OPPONENT_CHECKPOINT_TO_LOAD: str | None = None
# Relative path to a model checkpoint to load for the opponent before training (e.g. "checkpoints/actor_critic_episode_10.pth").
# Set to None to initialize the opponent with random weights.
OPPONENT_QUEUE_SIZE = 3 # Number of past opponents to keep in the queue for the agent to play against
# This helps prevent cyclical dynamics and encourages a more robust policy
# The opponent is sampled uniformly from the queue at the start of each episode
RANDOM_SEED = 42
TOURNAMENT_LENGTH = 5 # Number of episodes in a tournament (best of N)
# A tournament is a series of episodes where the agent competes against the same opponent queue
# If the agent wins the tournament it enters the opponent queue
SAVE_DIR = "checkpoints/ConvNeXt_actor_critic" # Relative path to save model checkpoints to during training
SAVE_EVERY_N = 10
SHOW_ENV = False # Whether to render the environment during training
VISUALIZATION_MODE = False # Whether or not to just play the agent against itself and not run training
# CHECKPOINT_TO_LOAD should be set for this to be interesting (otherwise the agent will just be randomly initialized)
# This works best if OPPONENT_CHECKPOINT_TO_LOAD is set otherwise the agent should just beat up the randomly initialized opponent
# The agent will be the one on the right side of the screen and the opponent will be on the left side


class GameStorage:
    def __init__(self):
        self.action_probs: list[torch.Tensor] = []
        self.action: list[int] = []
        self.rewards: list[float] = []
        self.state_values: list[torch.Tensor] = []


def normalize_observation(observation, device) -> torch.Tensor:
    """
    The observation in this environment is 4 stacked 210x160 grayscale images (210, 160, 4) numpy.ndarray.
    We normalize the pixel values to [0, 1] and permute to (4, 210, 160).
    """
    observation = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1) # Permute to [C, H, W]
    observation = observation.to(device)
    observation = observation / 255.0
    return observation


def update_agent(actor_critic: ActorCritic, optimizer: torch.optim.Optimizer, game_storage: GameStorage, device: torch.device):
    game_length = len(game_storage.state_values)

    # Extract scalar values for advantage computation
    values = game_storage.state_values

    # Compute GAE advantages working backwards
    advantages = torch.zeros(game_length, device=device)
    gae = 0.0
    for t in reversed(range(game_length)):
        next_value = values[t + 1].item() if t + 1 < game_length else 0.0
        delta = game_storage.rewards[t + 1] + GAMMA * next_value - values[t].item() # TD error
        gae = delta + GAMMA * GAE_LAMBDA * gae
        advantages[t] = gae

    returns = advantages + torch.tensor([v.item() for v in values], device=device)

    # Normalize advantages for stable training
    if game_length > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    entropies = []
    log_probs = []
    for i in range(game_length):
        entropy = torch.distributions.Categorical(game_storage.action_probs[i]).entropy()
        entropies.append(entropy)
        log_probs.append(torch.log(game_storage.action_probs[i][game_storage.action[i]]))
    entropies = torch.stack(entropies)
    log_probs = torch.stack(log_probs)
    stacked_values = torch.stack(values)

    actor_loss = -(log_probs * advantages.detach()).sum() - ENTROPY_COEFFICIENT * entropies.sum()
    critic_loss = F.mse_loss(stacked_values, returns.detach())
    total_loss = actor_loss + critic_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=MAX_GRAD_NORM) # Gradient clipping for stability
    optimizer.step()


def training_episode(actor_critic: ActorCritic, opponent: ActorCritic, optimizer: torch.optim.Optimizer,
                     device: torch.device, env, random_seed=None) -> tuple[int, int, int]:
    """
    Returns:
        tuple[int, int, int]: A tuple containing the final scores of the agent and opponent at the end of the episode, and the number of steps in the episode
    """
    observations, info = env.reset(seed=random_seed)
    
    game_storage = GameStorage()
    game_storage.rewards.append(0.0) # Add initial reward for the starting state (which is 0)
    
    agent_score = 0
    opponent_score = 0
    episode_steps = 0
    while env.agents:
        
        actions = {}
        for agent in env.agents:
            if agent == "first_0":
                # The agent we are training
                observation = observations[agent]
                obs_tensor = normalize_observation(observation, device)
                
                # Generate action for the agent
                action_probs, state_value = actor_critic(obs_tensor)
                distribution = torch.distributions.Categorical(action_probs)
                action = distribution.sample()
                action = int(action.item())
                actions[agent] = action
                
                game_storage.action_probs.append(action_probs)
                game_storage.action.append(action)
                game_storage.state_values.append(state_value)
                
            elif agent == "second_0":
                # The opponent
                observation = observations[agent]
                obs_tensor = normalize_observation(observation, device)
                obs_tensor = torch.flip(obs_tensor, [2])
                # Flip the observation horizontally for the opponent to maintain
                # a consistent perspective for the policy
                
                # Generate action for the opponent
                with torch.no_grad():
                    action_probs, state_value = opponent(obs_tensor)
                distribution = torch.distributions.Categorical(action_probs)
                action = distribution.sample()
                actions[agent] = int(action.item())
            else:
                raise ValueError(f"Unexpected agent name: {agent}")
            
        observations, rewards, terminations, truncations, infos = env.step(actions)
        episode_steps += 1
        
        if rewards["first_0"] > 0:
            agent_score += rewards["first_0"]
        if rewards["second_0"] > 0:
            opponent_score += rewards["second_0"]
        
        game_storage.rewards.append(rewards["first_0"])
        # This is the reward given upon reaching state i
        # so for the same index in episode_storage, rewards[i] is the reward for the
        # transition from state_values[i - 1] to state_values[i].
        # action_log_probs[i] corresponds to the action taken in state_values[i] to reach state_values[i + 1]
        
        if rewards["first_0"] != 0:
            # A game has concluded so update the agent
            # Updating after every game allows for faster learning and less memory usage compared to waiting
            # until the end of the episode
            update_agent(actor_critic, optimizer, game_storage, device=device)
            game_storage = GameStorage() # Reset the game storage for the next game in the episode
            game_storage.rewards.append(0.0) # Add initial reward for the starting state of the next game (which is 0)
    
    # Update the policy if we have any remaining transitions
    # This may occur if the final game ends in a draw (truncation)
    if len(game_storage.state_values) > 0:
        update_agent(actor_critic, optimizer, game_storage, device=device)

    # Return whether the agent won the episode
    return agent_score, opponent_score, episode_steps


def train_agent(actor_critic: ActorCritic, optimizer: torch.optim.Optimizer, device: torch.device, num_episodes: int = 1,
                starting_episode : int= 1, episodes_per_tournament=TOURNAMENT_LENGTH, random_seed=None, render_mode=None):
    """
    The actor critic agent and optimizer are expected to already be on the given device
    """
    
    # Initialize the environment
    env = pong_v3.parallel_env(obs_type="grayscale_image", render_mode=render_mode)
    env = supersuit.frame_stack_v1(env, 4)
    # Stacks the last 4 frames to deal with Atari flickering and to give the agent a sense of motion.
    # The resulting observation shape is (210, 160, 4)
    env = supersuit.frame_skip_v0(env, 4)
    # Skip frames for faster processing and less control
    # standard gym uses 4 frame skip for Atari games, so we follow that convention here
    env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)
    # Probabilistically repeat the last action to introduce non-determinism to the environment
    # This avoids the agent learning to exploit deterministic action patterns and encourages more robust policies
    
    cpu_device = torch.device("cpu")
    # Create CPU device to store opponent models on to save GPU memory when they aren't in use

    # Create our opponent as another instance of the actor-critic network
    opponent_queue = deque(maxlen=OPPONENT_QUEUE_SIZE) # Queue to hold past opponents for self-play
    opponent = ActorCritic()
    if OPPONENT_CHECKPOINT_TO_LOAD is not None:
        opponent_checkpoint_path = Path(__file__).resolve().parent / OPPONENT_CHECKPOINT_TO_LOAD
        opponent.load_state_dict(torch.load(opponent_checkpoint_path)["model"])
        print(f"Loaded opponent model checkpoint from {opponent_checkpoint_path}")
    opponent_queue.append(opponent)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=Path(__file__).resolve().parent / SAVE_DIR / "logs")
    test_observations, test_info = env.reset(seed=random_seed)
    test_agent_observation = test_observations["first_0"]
    writer.add_graph(actor_critic, normalize_observation(test_agent_observation, device))
    writer.add_text("hyperparameters", f"GAMMA: {GAMMA}\nGAE_LAMBDA: {GAE_LAMBDA}\nLEARNING_RATE: {LEARNING_RATE}\nNUM_EPISODES: {NUM_EPISODES}\nRANDOM_SEED: {RANDOM_SEED}\nTOURNAMENT_LENGTH: {TOURNAMENT_LENGTH}")

    agent_tournament_wins = 0
    tournament_games = 0
    for episode in range(starting_episode, num_episodes + 1):
        print(f"Starting episode {episode}/{num_episodes}...")
        
        opponent = random.choice(opponent_queue)
        opponent.to(device)

        # Run the training episode
        agent_score, opponent_score, episode_steps = training_episode(actor_critic, opponent, optimizer, device, env, random_seed)
        if agent_score > opponent_score:
            agent_tournament_wins += 1
        tournament_games += 1
        print(f"Episode {episode} completed. Agent score: {agent_score}, Opponent score: {opponent_score}")
        print(f"Agent now has a record of {agent_tournament_wins} wins out of {tournament_games} games in the current tournament.")
        writer.add_scalar("episode_steps", episode_steps, episode)
        # We log episode steps as a proxy for agent quality since as the agent/opponent gets better games should tend to last longer
        
        opponent.to(cpu_device) # Move opponent back to CPU to save GPU memory
        # After each tournament, if the agent beat the opponent more than half the time, update the opponent to be the agent
        if tournament_games >= episodes_per_tournament:
            if len(opponent_queue) < OPPONENT_QUEUE_SIZE or agent_tournament_wins > episodes_per_tournament // 2:
                opponent = ActorCritic()
                opponent.load_state_dict(actor_critic.state_dict())
                opponent_queue.append(opponent)
                print(f"Agent has joined the opponent queue!")
            agent_tournament_wins = 0 # Reset the tournament wins counter
            tournament_games = 0 # Reset the tournament games counter
            
        if episode % SAVE_EVERY_N == 0:
            save_path = Path(__file__).resolve().parent / SAVE_DIR / f"actor_critic_episode_{episode}.pth"
            torch.save(
                {
                    "model": actor_critic.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, save_path)
            print(f"Saved model to {save_path}")
    
    env.close()
    writer.flush()
    writer.close()
    

def visualize_agent(actor_critic: ActorCritic, opponent: ActorCritic, random_seed=None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = pong_v3.parallel_env(obs_type="grayscale_image", render_mode="human")
    env = supersuit.frame_stack_v1(env, 4)
    env = supersuit.frame_skip_v0(env, 4)
    env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)
    observations, info = env.reset(seed=random_seed)

    actor_critic.to(device)
    opponent.to(device)
    
    agent_score = 0
    opponent_score = 0
    episode_steps = 1
    while env.agents:
        
        actions = {}
        for agent in env.agents:
            if agent == "first_0":
                # The agent we are training
                observation = observations[agent]
                obs_tensor = normalize_observation(observation, device)
                
                # Generate action for the agent
                action_probs, state_value = actor_critic(obs_tensor)
                distribution = torch.distributions.Categorical(action_probs)
                action = distribution.sample()
                action = int(action.item())
                actions[agent] = action
                
            elif agent == "second_0":
                # The opponent
                observation = observations[agent]
                obs_tensor = normalize_observation(observation, device)
                obs_tensor = torch.flip(obs_tensor, dims=[2])
                # Flip the observation horizontally for the opponent to maintain
                # a consistent perspective for the policy
                
                # Generate action for the opponent
                with torch.no_grad():
                    action_probs, state_value = opponent(obs_tensor)
                distribution = torch.distributions.Categorical(action_probs)
                action = distribution.sample()
                actions[agent] = int(action.item())
            else:
                raise ValueError(f"Unexpected agent name: {agent}")
            
        observations, rewards, terminations, truncations, infos = env.step(actions)
        episode_steps += 1
        
        if rewards["first_0"] > 0:
            agent_score += rewards["first_0"]
        if rewards["second_0"] > 0:
            opponent_score += rewards["second_0"]
    
    print("Game over!")
    print(f"Total episode steps: {episode_steps}")
    if agent_score > opponent_score:
        print(f"Agent wins! Final score - Agent: {agent_score}, Opponent: {opponent_score}")
    elif opponent_score > agent_score:
        print(f"Opponent wins! Final score - Agent: {agent_score}, Opponent: {opponent_score}")
    else:
        print(f"It's a draw! Final score - Agent: {agent_score}, Opponent: {opponent_score}")
    
    env.close()

    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = ActorCritic().to(device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=LEARNING_RATE)
    starting_episode = 1
    if CHECKPOINT_TO_LOAD is not None:
        import re
        
        checkpoint_path = Path(__file__).resolve().parent / CHECKPOINT_TO_LOAD
        episode_regex = r"actor_critic_episode_(\d+)\.pth"
        match = re.search(episode_regex, checkpoint_path.name)
        if match:
            starting_episode = int(match.group(1)) + 1
        else:
            raise ValueError(f"Checkpoint filename does not match expected format 'actor_critic_episode_X.pth': {checkpoint_path.name}")
        checkpoint = torch.load(checkpoint_path)
        agent.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Loaded model checkpoint from {checkpoint_path}")
        if not VISUALIZATION_MODE:
            print(f"Resuming training from episode {starting_episode - 1}...")
    if VISUALIZATION_MODE:
        opponent = ActorCritic()
        if OPPONENT_CHECKPOINT_TO_LOAD is not None:
            opponent_checkpoint_path = Path(__file__).resolve().parent / OPPONENT_CHECKPOINT_TO_LOAD
            opponent.load_state_dict(torch.load(opponent_checkpoint_path)["model"])
            print(f"Loaded opponent model checkpoint from {opponent_checkpoint_path}")
        visualize_agent(agent, opponent, random_seed=RANDOM_SEED)
    else:
        train_agent(agent, optimizer, device, num_episodes=NUM_EPISODES, starting_episode=starting_episode, random_seed=RANDOM_SEED, render_mode="human" if SHOW_ENV else None)