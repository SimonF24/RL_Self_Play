from collections import deque
from pathlib import Path
from pettingzoo.atari import pong_v3
import random
import supersuit
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from actor_critic import ActorCritic


# In general these hyperparameters try to follow the defaults in stable-baselines3
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py#L80

CHECKPOINT_TO_LOAD: str | None = None
# Relative path to a model checkpoint to load before training (e.g. "checkpoints/actor_critic_episode_10.pth").
# Set to None to train from scratch.
ENTROPY_COEFFICIENT = 0.01 # Coefficient for entropy regularization to encourage exploration
GAMMA = 0.99 # Discount factor for future rewards/state values
GAE_LAMBDA = 0.95 # Lambda parameter for GAE
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 0.5 # Max gradient norm for clipping to improve training stability
NUM_EPISODES = 4000
# Each episode is one game of first to 21 points
NUM_EPOCHS: int = 10
# Numbers of epochs to train for on each point
# An epoch is a single pass through the transitions collected from a point
OPPONENT_CHECKPOINT_TO_LOAD: str | None = None
# Relative path to a model checkpoint to load for the opponent before training (e.g. "checkpoints/actor_critic_episode_10.pth").
# Set to None to initialize the opponent with random weights.``
OPPONENT_QUEUE_SIZE = 3 # Number of past opponents to keep in the queue for the agent to play against
# This helps prevent cyclical dynamics and encourages a more robust policy
# The opponent is sampled uniformly from the queue at the start of each episode
POINTS_PER_UPDATE = 5 # Number of points to play before updating the agent
# Setting this too low results is noisy advantage normalization that can impair learning
# Setting this high enough will train once per match (episode)
# This this too high might result in running out of memory once the agent gets better and 
# points go longer
PPO_CLIP_EPSILON = 0.2 # Clipping epsilon for PPO surrogate objective to limit policy updates and improve stability
RANDOM_SEED = 42
RECORD_VIDEO = False # Record a video of one episode via the visualize_agent pipeline without displaying the screen
# Requires imageio and imageio-ffmpeg: pip install imageio imageio-ffmpeg
# Videos are saved to SAVE_DIR
TOURNAMENT_LENGTH = 5 # Number of episodes in a tournament (best of N)
# A tournament is a series of episodes where the agent competes against the same opponent queue
# If the agent wins the tournament it enters the opponent queue
SAVE_DIR = "checkpoints/ConvNeXt" # Relative path to save model checkpoints to during training
SAVE_EVERY_N = 10
SHOW_ENV = False # Whether to render the environment during training
VALUE_COEFFICIENT = 0.5 # Multiplicative coefficient for value loss
VISUALIZATION_MODE = False # Whether or not to just play the agent against itself and not run training
# CHECKPOINT_TO_LOAD should be set for this to be interesting (otherwise the agent will just be randomly initialized)
# This works best if OPPONENT_CHECKPOINT_TO_LOAD is set otherwise the agent should just beat up the randomly initialized opponent
# The agent will be the one on the right side of the screen and the opponent will be on the left side


# This is the mapping for our reduced action space (removing the fire actions)
action_map = {
    0: 0, # NOOP
    1: 2, # Move right (up)
    2: 3, # Move left (down)
}
serve_actions = [1, 4, 5] # Fire, Fire right, Fire left


class GameStorage:
    def __init__(self):
        self.action: list[int] = []
        self.action_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.states: list[torch.Tensor] = []
        self.state_values: list[torch.Tensor] = []
        

def normalize_observation(observation, device) -> torch.Tensor:
    """
    The observation in this environment is 4 stacked 210x160 grayscale images (210, 160, 4) numpy.ndarray.
    We normalize the pixel values to [0, 1] and permute to (4, 210, 160).
    """
    observation = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1) # Permute to [C, H, W]
    observation = observation.to(device)
    observation = observation.unsqueeze(0)
    # observation = F.interpolate(observation, size=(84, 84), mode="bilinear")
    observation = observation / 255.0
    return observation


def wrap_environment(env):
    """Wraps the environment with preprocessing functions"""
    env = supersuit.frame_stack_v1(env, 4)
    # Stacks the last 4 frames to deal with Atari flickering and to give the agent a sense of motion.
    # The resulting observation shape is (210, 160, 4)
    env = supersuit.frame_skip_v0(env, 4)
    # Skip frames for faster processing and less control
    # standard gym uses 4 frame skip for Atari games, so we follow that convention here to enable pretraining
    env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)
    # Probabilistically repeat the last action to introduce non-determinism to the environment
    # This avoids the agent learning to exploit deterministic action patterns and encourages more robust policies
    # This is also used in standard gym
    return env


def update_agent(actor_critic: ActorCritic, optimizer: torch.optim.Optimizer, game_storage: GameStorage,
                 device: torch.device, writer: SummaryWriter, episode: int) -> None:
    game_length = len(game_storage.states)
    
    # Store original state values and action probabilities from the game 
    original_action_probs = torch.zeros(game_length, device=device)
    original_state_values = torch.zeros(game_length, device=device)
    for j in range(game_length):
        original_action_probs[j] = game_storage.action_probs[j][game_storage.action[j]].detach()
        original_state_values[j] = game_storage.state_values[j].detach()
    
    # Compute GAE advantages working backwards
    advantages = torch.zeros(game_length, device=device)
    gae = 0.0
    for j in reversed(range(game_length)):
        next_value = game_storage.state_values[j + 1].item() if j + 1 < game_length else 0.0
        delta = game_storage.rewards[j + 1] + GAMMA * next_value - game_storage.state_values[j].item() # TD error
        gae = delta + GAMMA * GAE_LAMBDA * gae
        advantages[j] = gae
        
    returns = (original_state_values + advantages).detach()
    
    # Normalize advantages for stability
    if game_length > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    episode_entropy_loss = 0.0
    episode_surrogate_loss = 0.0
    episode_total_loss = 0.0
    episode_value_loss = 0.0
    for i in range(NUM_EPOCHS):
        
        states = torch.cat(game_storage.states, dim=0)
        action_probs, new_state_values = actor_critic(states)
        entropies = torch.zeros(game_length, device=device)
        new_action_probs = torch.zeros(game_length, device=device)
        for j in range(len(game_storage.states)):
            entropies[j] = torch.distributions.Categorical(action_probs[j, :]).entropy()
            new_action_probs[j] = action_probs[j, game_storage.action[j]]
        
        # PPO surrogate loss with clipping
        ratio = torch.exp(torch.log(new_action_probs + 1e-8) - torch.log(original_action_probs + 1e-8))
        # The above is division in log space for numerical stability
        clipped_ratio = torch.clamp(ratio, 1 - PPO_CLIP_EPSILON, 1 + PPO_CLIP_EPSILON)
        surrogate_loss = -torch.min(ratio * advantages.detach(), clipped_ratio * advantages.detach()).mean()
        value_loss = F.mse_loss(returns, new_state_values)
        entropy_loss = -entropies.mean()
        total_loss = surrogate_loss + VALUE_COEFFICIENT * value_loss + ENTROPY_COEFFICIENT * entropy_loss

        episode_entropy_loss += ENTROPY_COEFFICIENT * entropy_loss.item()
        episode_surrogate_loss += surrogate_loss.item()
        episode_total_loss += total_loss.item()
        episode_value_loss += VALUE_COEFFICIENT * value_loss.item()
        
        # Perform the optimization step
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=MAX_GRAD_NORM)
        # Gradient clipping for stability
        optimizer.step()
        
    # Log losses to TensorBoard
    writer.add_scalar("loss/surrogate_loss", episode_surrogate_loss, episode)
    writer.add_scalar("loss/value_loss", episode_value_loss, episode)
    writer.add_scalar("loss/entropy_loss", episode_entropy_loss, episode)
    writer.add_scalar("loss/total_loss", episode_total_loss, episode)


def training_episode(actor_critic: ActorCritic, opponent: ActorCritic, optimizer: torch.optim.Optimizer,
                     device: torch.device, env, writer: SummaryWriter, episode: int, random_seed=None
                     ) -> tuple[int, int, int]:
    """
    Returns:
        tuple[int, int, int]: A tuple containing the final scores of the agent and opponent at the end of the episode, and the number of steps in the episode
    """
    observations, infos = env.reset(seed=random_seed)
    
    game_storage = GameStorage()
    game_storage.rewards.append(0.0) # Add initial reward for the starting state (which is 0)
    
    agent_score = 0
    opponent_score = 0
    episode_steps = 0
    games_stored = 0
    serve_flag = True
    while env.agents:
        
        # Auto-serve
        if serve_flag:
            # Make each agent randomly choose its serve action
            actions = {
                "first_0": random.choice(serve_actions),
                "second_0": random.choice(serve_actions)
            }
            # We repeat the serve action to make sure it actually goes through and the game starts
            for _ in range(16):
                observations, rewards, terminations, truncations, infos = env.step(actions)
                episode_steps += 1
            serve_flag = False
        
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
                actions[agent] = action_map[action]
                
                game_storage.action_probs.append(action_probs)
                game_storage.action.append(action)
                game_storage.states.append(obs_tensor)
                game_storage.state_values.append(state_value)
                
            elif agent == "second_0":
                # The opponent
                observation = observations[agent]
                obs_tensor = normalize_observation(observation, device)
                obs_tensor = torch.flip(obs_tensor, [3])
                # Flip the observation horizontally for the opponent to maintain
                # a consistent perspective for the policy
                
                # Generate action for the opponent
                with torch.no_grad():
                    action_probs, state_value = opponent(obs_tensor)
                distribution = torch.distributions.Categorical(action_probs)
                action = distribution.sample()
                actions[agent] = action_map[int(action.item())]
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
            serve_flag = True
            games_stored += 1
            if games_stored % POINTS_PER_UPDATE == 0:
                update_agent(actor_critic, optimizer, game_storage, device, writer, episode)
                game_storage = GameStorage()
                # Reset the game storage for the next game in the episode
                game_storage.rewards.append(0.0)
                # Add initial reward for the starting state of the next game (which is 0)
    
    # Update the policy if we have any remaining transitions
    # This may occur if the final game ends in a draw (truncation)
    if len(game_storage.state_values) > 0:
        update_agent(actor_critic, optimizer, game_storage, device, writer, episode)

    return agent_score, opponent_score, episode_steps


def train_agent(actor_critic: ActorCritic, optimizer: torch.optim.Optimizer, device: torch.device, num_episodes: int,
                starting_episode: int = 1, episodes_per_tournament=TOURNAMENT_LENGTH,
                random_seed=None, render_mode=None):
    """
    The actor critic agent and optimizer are expected to already be on the given device
    """
    
    # Initialize the environment
    env = pong_v3.parallel_env(obs_type="grayscale_image", render_mode=render_mode)
    env = wrap_environment(env)
    
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
    writer.add_text(
        "hyperparameters",
        f"GAMMA: {GAMMA}\nGAE_LAMBDA: {GAE_LAMBDA}\nLEARNING_RATE: {LEARNING_RATE}"
        "\nNUM_EPISODES: {NUM_EPISODES}\nRANDOM_SEED: {RANDOM_SEED}\nTOURNAMENT_LENGTH: {TOURNAMENT_LENGTH}"
        "\nOPPONENT_QUEUE_SIZE: {OPPONENT_QUEUE_SIZE}\nPPO_CLIP_EPSILON: {PPO_CLIP_EPSILON}"
        "\nENTROPY_COEFFICIENT: {ENTROPY_COEFFICIENT}\nNUM_EPOCHS: {NUM_EPOCHS}"
        "\nMAX_GRAD_NORM: {MAX_GRAD_NORM}"
    )

    agent_tournament_wins = 0
    tournament_games = 0
    for episode in range(starting_episode, num_episodes + 1):
        print(f"Starting episode {episode}/{num_episodes}...")
        
        opponent = random.choice(opponent_queue)
        opponent.to(device)

        # Run the training episode
        agent_score, opponent_score, episode_steps = training_episode(
            actor_critic, opponent, optimizer, device, env, writer, episode, random_seed
        )
        if agent_score > opponent_score:
            agent_tournament_wins += 1
        tournament_games += 1
        print(f"Episode {episode} completed. Agent score: {agent_score}, Opponent score: {opponent_score}")
        print(f"Agent now has a record of {agent_tournament_wins} wins out of {tournament_games} games in the current tournament.")
        writer.add_scalar("episode_steps", episode_steps, episode)
        # We log episode steps as a proxy for agent quality since as the agent/opponent gets better games should tend to last longer
        writer.add_scalar("agent_score", agent_score, episode)
        writer.add_scalar("opponent_score", opponent_score, episode)
        
        opponent.to(cpu_device) # Move opponent back to CPU to save GPU memory
        # After each tournament, if the agent beat the opponent more than half the time
        # update the opponent queue to include the agent
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
    

def visualize_agent(actor_critic: ActorCritic, opponent: ActorCritic, device: torch.device,
                    random_seed=None, record_video=False) -> None:
    render_mode = "rgb_array" if record_video else "human"
    env = pong_v3.parallel_env(obs_type="grayscale_image", render_mode=render_mode)
    env = wrap_environment(env)
    observations, info = env.reset(seed=random_seed)

    actor_critic.to(device)
    opponent.to(device)

    frames = [] if record_video else None

    agent_score = 0
    opponent_score = 0
    episode_steps = 1
    serve_flag = True # Flag to indicate whether we should serve to start a game
    while env.agents:

        if serve_flag:
            # Both agents take a random serve (fire) action to start the game
            actions = {
                "first_0": random.choice(serve_actions),
                "second_0": random.choice(serve_actions)
            }
            # We repeat the serve action to make sure it actually goes through
            for _ in range(16):
                observations, rewards, terminations, truncations, infos = env.step(actions)
                episode_steps += 1
                if record_video:
                    frames.append(env.render())
            serve_flag = False

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
                actions[agent] = action_map[int(action.item())]

            elif agent == "second_0":
                # The opponent
                observation = observations[agent]
                obs_tensor = normalize_observation(observation, device)
                obs_tensor = torch.flip(obs_tensor, dims=[3])
                # Flip the observation horizontally for the opponent to maintain
                # a consistent perspective for the policy

                # Generate action for the opponent
                with torch.no_grad():
                    action_probs, state_value = opponent(obs_tensor)
                distribution = torch.distributions.Categorical(action_probs)
                action = distribution.sample()
                actions[agent] = action_map[int(action.item())]
            else:
                raise ValueError(f"Unexpected agent name: {agent}")

        observations, rewards, terminations, truncations, infos = env.step(actions)
        episode_steps += 1
        if record_video:
            frames.append(env.render())

        if rewards["first_0"] > 0:
            agent_score += rewards["first_0"]
            serve_flag = True
        if rewards["second_0"] > 0:
            opponent_score += rewards["second_0"]
            serve_flag = True

    print("Game over!")
    print(f"Total episode steps: {episode_steps}")
    if agent_score > opponent_score:
        print(f"Agent wins! Final score - Agent: {agent_score}, Opponent: {opponent_score}")
    elif opponent_score > agent_score:
        print(f"Opponent wins! Final score - Agent: {agent_score}, Opponent: {opponent_score}")
    else:
        print(f"It's a draw! Final score - Agent: {agent_score}, Opponent: {opponent_score}")

    env.close()

    if record_video and frames:
        import imageio
        video_path = Path(__file__).resolve().parent / SAVE_DIR / "agent_video.mp4"
        imageio.mimsave(str(video_path), frames, fps=30)
        print(f"Saved video to {video_path}")

    
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
        if not (RECORD_VIDEO or VISUALIZATION_MODE):
            print(f"Resuming training from episode {starting_episode - 1}...")
    if RECORD_VIDEO or VISUALIZATION_MODE:
        opponent = ActorCritic()
        if OPPONENT_CHECKPOINT_TO_LOAD is not None:
            opponent_checkpoint_path = Path(__file__).resolve().parent / OPPONENT_CHECKPOINT_TO_LOAD
            opponent.load_state_dict(torch.load(opponent_checkpoint_path)["model"])
            print(f"Loaded opponent model checkpoint from {opponent_checkpoint_path}")
        visualize_agent(agent, opponent, device, random_seed=RANDOM_SEED, record_video=RECORD_VIDEO)
    else:
        train_agent(agent, optimizer, device, NUM_EPISODES, starting_episode=starting_episode,
                    random_seed=RANDOM_SEED, render_mode="human" if SHOW_ENV else None)