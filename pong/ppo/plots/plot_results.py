import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

pretraining_episode_length = pd.read_csv("data/pretraining_episode_length.csv")
pretraining_agent_score = pd.read_csv("data/pretraining_agent_score.csv")
training_episode_length = pd.read_csv("data/training_episode_length.csv")
training_agent_score = pd.read_csv("data/training_agent_score.csv")

fig = plt.figure(figsize=(14, 10))
fig.suptitle("PPO Training Results", fontsize=16, fontweight="bold")
gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(
    pretraining_episode_length["Step"],
    pretraining_episode_length["Value"],
    alpha=0.4,
    color="steelblue",
    linewidth=0.8,
    label="Raw"
)
ax1.plot(
    pretraining_episode_length["Step"],
    pretraining_episode_length["Value"].rolling(20, min_periods=1).mean(),
    color="steelblue",
    linewidth=2,
    label="20-step rolling mean"
)
ax1.set_title("Pretraining — Episode Length")
ax1.set_xlabel("Step")
ax1.set_ylabel("Steps per Episode")

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(
    pretraining_agent_score["Step"],
    pretraining_agent_score["Value"],
    alpha=0.4,
    color="darkorange",
    linewidth=0.8
)
ax2.plot(
    pretraining_agent_score["Step"],
    pretraining_agent_score["Value"].rolling(20, min_periods=1).mean(),
    color="darkorange",
    linewidth=2
)
ax2.set_title("Pretraining — Agent Score")
ax2.set_xlabel("Step")
ax2.set_ylabel("Score")

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(
    training_episode_length["Step"],
    training_episode_length["Value"],
    alpha=0.4,
    color="steelblue",
    linewidth=0.8
)
ax3.plot(
    training_episode_length["Step"],
    training_episode_length["Value"].rolling(20, min_periods=1).mean(),
    color="steelblue",
    linewidth=2
)
ax3.set_title("Self-Play Training — Episode Length")
ax3.set_xlabel("Step")
ax3.set_ylabel("Steps per Episode")

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(
    training_agent_score["Step"],
    training_agent_score["Value"],
    alpha=0.4,
    color="darkorange",
    linewidth=0.8
)
ax4.plot(
    training_agent_score["Step"],
    training_agent_score["Value"].rolling(20, min_periods=1).mean(),
    color="darkorange",
    linewidth=2
)
ax4.set_title("Self-Play Training — Agent Score")
ax4.set_xlabel("Step")
ax4.set_ylabel("Score")

for ax in [ax1, ax2, ax3, ax4]:
    ax.grid(True, alpha=0.3)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.02), frameon=True)
fig.subplots_adjust(bottom=0.1)

plt.savefig("ppo_results.png", dpi=150, bbox_inches="tight")
print("Saved ppo_results.png")
plt.show()
