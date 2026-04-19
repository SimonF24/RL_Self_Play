import ale_py
import gymnasium as gym
from pathlib import Path
import random
import supersuit
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from actor_critic import ActorCritic
from training import action_map, GameStorage, normalize_observation, serve_actions

# In general these hyperparameters try to follow the defaults in stable-baselines3
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py#L80

CHECKPOINT_TO_LOAD: str | None = None
DIFFICULTY: int = 0 # 0-3 This controls the size of the paddles, 0 is biggest, 3 is smallest
# 0 matches the multi-agent environment
ENTROPY_COEFFICIENT = 0.02 # Coefficient for entropy regularization to encourage exploration
GAE_LAMBDA = 0.95 # Lambda parameter for GAE
GAME_MODE = 0 # 0 or 1
GAMMA = 0.99 # Discount factor for future rewards/state values
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 0.5 # Max gradient norm for clipping to improve training stability
NUM_EPISODES = 3000
# Each episode is one game of first to 21 points
NUM_EPOCHS: int = 10
# Numbers of epochs to train for on each point
# An epoch is a single pass through the transitions collected from a point
POINTS_PER_UPDATE = 5 # Number of points to play before updating the agent
# Setting this too low results is noisy advantage normalization that can impair learning
# Setting this high enough will train once per match (episode)
# This this too high might result in running out of memory once the agent gets better and 
# points go longer
PPO_CLIP_EPSILON = 0.2 # Clipping epsilon for PPO surrogate objective to limit policy updates and improve stability
RANDOM_SEED = 42
RECORD_VIDEO = False # Record a video, if True the same pipeline as VISUALIZATION_MODE will be run
SAVE_DIR = "checkpoints/pretraining_ConvNeXt" # Relative path to save model checkpoints to during training
SAVE_EVERY_N = 10
SHOW_ENV = False
VALUE_COEFFICIENT = 0.5 # Multiplicative coefficient for value loss
VISUALIZATION_MODE = False # Whether or not to just play the agent against itself and not run training
# CHECKPOINT_TO_LOAD should be set for this to be interesting (otherwise the agent will just be randomly initialized)

# This is a pretraining function where we play against the Atari built-in opponent.
# This can be a useful baseline for comparing our self-play trained models to or evaluating our
# self-play trained models against

def wrap_environment(env):
    """
    Wraps the environment with preprocessing functions
    Unlike the multi-agent version used in self-play stickiness
    and frame skipping are already provided in the environment by default
    """
    env = supersuit.frame_stack_v1(env, 4)
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
    
    
def training_episode(actor_critic: ActorCritic, optimizer: torch.optim.Optimizer, device: torch.device,
                     env, writer: SummaryWriter, episode: int, random_seed=None):
    
    observation, info = env.reset(seed=random_seed)
    
    game_storage = GameStorage()
    game_storage.rewards.append(0.0) # Add initial reward for the starting state (which is 0)
    
    agent_score = 0
    opponent_score = 0
    episode_steps = 0
    games_stored = 0
    serve_flag = True # The agent serves first, after that the winner of the last point serves
    terminated = False
    truncated = False
    while not (terminated or truncated):
        
        # Automatic serving so the agent doesn't have to learn it and we can cut down the action space
        if serve_flag:
            # Agent takes a random serve (fire) action to start the game
            action = random.choice(serve_actions)
            # We repeat the serve action to make it actually go through
            for _ in range(16):
                observation, reward, terminated, truncated, info = env.step(action)
                episode_steps += 1
            serve_flag = False
            
        # Generate action for the agent
        obs_tensor = normalize_observation(observation, device)
        action_probs, state_value = actor_critic(obs_tensor)
        action_probs = action_probs.squeeze(0)
        state_value = state_value.squeeze(0)
        distribution = torch.distributions.Categorical(action_probs)
        action = distribution.sample()
        action = int(action.item())
        env_action = action_map[action]
        
        game_storage.action_probs.append(action_probs)
        game_storage.action.append(action)
        game_storage.states.append(obs_tensor)
        game_storage.state_values.append(state_value)
        
        observation, reward, terminated, truncated, info = env.step(env_action)
        episode_steps += 1
        
        game_storage.rewards.append(reward)
        # This is the reward given upon reaching state i
        # so for the same index in episode_storage, rewards[i] is the reward for the
        # transition from state_values[i - 1] to state_values[i].
        # action_log_probs[i] corresponds to the action taken in state_values[i] to reach state_values[i + 1]
        
        if reward > 0:
            agent_score += 1
            serve_flag = True
        elif reward < 0:
            opponent_score += 1
        
        if reward != 0:
            games_stored += 1
            if games_stored % POINTS_PER_UPDATE == 0:
                update_agent(actor_critic, optimizer, game_storage, device, writer, episode)
                game_storage = GameStorage()
                # Reset the game storage for the next game in the episode
                game_storage.rewards.append(0.0)
                # Add initial reward for the starting state of the next game (which is 0)
                games_stored = 0
        
    # Update the policy if we have any remaining transitions
    # This may occur if the final game ends in a draw (truncation)
    if len(game_storage.state_values) > 0:
        update_agent(actor_critic, optimizer, game_storage, device, writer, episode)
    
    return agent_score, opponent_score, episode_steps


def train_agent(actor_critic: ActorCritic, optimizer: torch.optim.Optimizer, device: torch.device,
                num_episodes: int, starting_episode: int, random_seed=None, render_mode=None) -> None:
    env = gym.make("ALE/Pong-v5", difficulty=DIFFICULTY, obs_type="grayscale", render_mode=render_mode)
    env = wrap_environment(env)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=Path(__file__).resolve().parent / SAVE_DIR / "logs")
    test_observation, test_info = env.reset(seed=random_seed)
    writer.add_graph(actor_critic, normalize_observation(test_observation, device))
    writer.add_text(
        "hyperparameters",
        f"GAMMA: {GAMMA}\nGAE_LAMBDA: {GAE_LAMBDA}\nLEARNING_RATE: {LEARNING_RATE}"
        "\nNUM_EPISODES: {NUM_EPISODES}\nRANDOM_SEED: {RANDOM_SEED}\n"
        "\n\nPPO_CLIP_EPSILON: {PPO_CLIP_EPSILON}\nENTROPY_COEFFICIENT: {ENTROPY_COEFFICIENT}"
        "\nNUM_EPOCHS: {NUM_EPOCHS}\nMAX_GRAD_NORM: {MAX_GRAD_NORM}"
    )
    
    for episode in range(starting_episode, num_episodes + 1):
        print(f"Starting episode {episode}/{num_episodes}...")
        
        agent_score, opponent_score, episode_steps = training_episode(
            actor_critic, optimizer, device, env, writer, episode, random_seed)
        print(f"Episode {episode} completed. Agent score: {agent_score}, Opponent score: {opponent_score}")
        writer.add_scalar("episode_steps", episode_steps, episode)
        writer.add_scalar("agent_score", agent_score, episode)
        writer.add_scalar("opponent_score", opponent_score, episode)

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

    return None


def visualize_agent(actor_critic: ActorCritic, device: torch.device, random_seed=None) -> None:
    env = gym.make("ALE/Pong-v5",
                   difficulty=DIFFICULTY,
                   obs_type="grayscale",
                   render_mode="rgb_array" if RECORD_VIDEO else "human")
    env = wrap_environment(env)
    if RECORD_VIDEO:
        env = gym.wrappers.RecordVideo(env, video_folder=str(Path(__file__).resolve().parent / SAVE_DIR))
    observation, info = env.reset(seed=random_seed)
    terminated = False
    truncated = False

    actor_critic.to(device)
    
    agent_score = 0
    opponent_score = 0
    episode_steps = 1
    serve_flag = True # Flag to indicate whether we should serve to start a game
    while not (terminated or truncated):
        
        # Automatic serving
        if serve_flag:
            # Agent takes a random serve (fire) action to start the game
            action = random.choice(serve_actions)
            # We repeat the serve action to make it actually go through
            for _ in range(16):
                observation, reward, terminated, truncated, info = env.step(action)
                episode_steps += 1
            serve_flag = False
        
        # Generate action for the agent
        obs_tensor = normalize_observation(observation, device)
        action_probs, state_value = actor_critic(obs_tensor)
        action_probs = action_probs.squeeze(0)
        distribution = torch.distributions.Categorical(action_probs)
        action = distribution.sample()
        action = action_map[int(action.item())]
            
        observation, reward, terminated, truncated, info = env.step(action)
        episode_steps += 1
        
        if reward > 0:
            agent_score += 1
            serve_flag = True
        elif reward < 0:
            opponent_score += 1
    
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
    gym.register_envs(ale_py)
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
    if VISUALIZATION_MODE or RECORD_VIDEO:
        visualize_agent(agent, device, RANDOM_SEED)
    else:
        train_agent(agent, optimizer, device, NUM_EPISODES, starting_episode,
                    random_seed=RANDOM_SEED, render_mode="human" if SHOW_ENV else None)