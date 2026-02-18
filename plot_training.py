import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.facecolor": "#f7f9fc",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#d6dce5",
        "axes.labelcolor": "#25364d",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "grid.color": "#dfe5ee",
        "grid.alpha": 0.7,
        "font.size": 10,
    }
)


def tensor_to_float(series: pd.Series) -> pd.Series:
    # Converts values like "tensor(1.23, grad_fn=<...>)" to 1.23
    extracted = series.astype(str).str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0]
    return pd.to_numeric(extracted, errors="coerce")


df = pd.read_csv("training_metrics.csv")

for col in [
    "episode",
    "total_reward",
    "policy_loss",
    "value_loss",
    "entropy",
    "final_quality",
    "dev_calls",
    "analyst_calls",
    "reviewer_calls",
    "tester_calls",
]:
    if col in df.columns:
        df[col] = tensor_to_float(df[col])

df = df.dropna(subset=["episode"])

# Moving average
df["reward_ma"] = df["total_reward"].rolling(window=20).mean()

fig, axes = plt.subplots(3, 2, figsize=(13, 11), dpi=120)
fig.suptitle("Training Metrics Overview", fontsize=15, fontweight="bold", color="#1f2a44")

axes[0, 0].plot(df["episode"], df["total_reward"], color="#7aa6ff", alpha=0.28, lw=1.2, label="reward")
axes[0, 0].plot(df["episode"], df["reward_ma"], color="#1f5fe0", lw=2.2, label="moving avg")
axes[0, 0].set_title("Episode Reward")
axes[0, 0].set_xlabel("Episode")
axes[0, 0].set_ylabel("Reward")
axes[0, 0].legend(frameon=False)

axes[0, 1].plot(df["episode"], df["policy_loss"], label="Policy Loss", color="#e76f51", lw=2.0)
axes[0, 1].plot(df["episode"], df["value_loss"], label="Value Loss", color="#f4a261", lw=2.0)
axes[0, 1].legend(frameon=False)
axes[0, 1].set_title("Losses")
axes[0, 1].set_xlabel("Episode")
axes[0, 1].set_ylabel("Loss")

axes[1, 0].plot(df["episode"], df["entropy"], color="#2a9d8f", lw=2.0)
axes[1, 0].set_title("Entropy")
axes[1, 0].set_xlabel("Episode")
axes[1, 0].set_ylabel("Entropy")

axes[1, 1].plot(df["episode"], df["final_quality"], color="#6f42c1", lw=2.0)
axes[1, 1].set_title("Final Quality")
axes[1, 1].set_xlabel("Episode")
axes[1, 1].set_ylabel("Quality")

if "dev_calls" in df.columns:
    axes[2, 0].plot(df["episode"], df["dev_calls"], label="Dev Calls", color="#457b9d", lw=2.0)
if "analyst_calls" in df.columns:
    axes[2, 0].plot(df["episode"], df["analyst_calls"], label="Analyst Calls", color="#1d3557", lw=2.0)
if "reviewer_calls" in df.columns:
    axes[2, 0].plot(df["episode"], df["reviewer_calls"], label="Reviewer Calls", color="#e63946", lw=2.0)
if "tester_calls" in df.columns:
    axes[2, 0].plot(df["episode"], df["tester_calls"], label="Tester Calls", color="#ffb703", lw=2.0)
axes[2, 0].set_title("Nb Calls Evolution")
axes[2, 0].set_xlabel("Episode")
axes[2, 0].set_ylabel("Calls")
axes[2, 0].legend(frameon=False)

axes[2, 1].axis("off")

for ax in axes.flat:
    ax.grid(True, linestyle="-", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
