import glob
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Setup clean publication style
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#d1d5db"
plt.rcParams["axes.linewidth"] = 1.0

def find_trainer_state_or_logs():
    """Tự động tìm kiếm file trainer_state.json hoặc logging.jsonl thật từ server."""
    possible_paths = glob.glob("artifacts/training/trainer_state_final.json") + \
                     glob.glob("artifacts/training/**/trainer_state.json", recursive=True) + \
                     glob.glob("training/**/trainer_state.json", recursive=True) + \
                     glob.glob("**/trainer_state.json", recursive=True) + \
                     glob.glob("artifacts/training/**/logging.jsonl", recursive=True) + \
                     glob.glob("training/**/logging.jsonl", recursive=True) + \
                     glob.glob("**/logging.jsonl", recursive=True)
    for p in possible_paths:
        if Path(p).exists() and Path(p).is_file():
            return Path(p)
    return None

def extract_real_metrics(log_file):
    steps = []
    losses = []
    eval_steps = []
    eval_losses = []
    
    if not log_file or not Path(log_file).exists():
        return steps, losses, eval_steps, eval_losses

    path = Path(log_file)

    if path.name.startswith("trainer_state") or path.suffix == ".json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            log_history = data.get("log_history", [])
            for entry in log_history:
                if "loss" in entry and "step" in entry:
                    steps.append(entry["step"])
                    losses.append(entry["loss"])
                if "eval_loss" in entry and "step" in entry:
                    eval_steps.append(entry["step"])
                    eval_losses.append(entry["eval_loss"])
        except Exception as e:
            print(f"Error parsing JSON: {e}")
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    step = entry.get("step")
                    if step is None and "global_step/max_steps" in entry:
                        step = int(str(entry["global_step/max_steps"]).split("/", 1)[0])
                    if "loss" in entry and step is not None:
                        steps.append(step)
                        losses.append(entry["loss"])
                    if "eval_loss" in entry and step is not None:
                        eval_steps.append(step)
                        eval_losses.append(entry["eval_loss"])
                except json.JSONDecodeError:
                    match = re.search(r"step:\s*(\d+).*?loss:\s*([0-9\.]+)", line, re.IGNORECASE)
                    if match:
                        steps.append(int(match.group(1)))
                        losses.append(float(match.group(2)))

    return steps, losses, eval_steps, eval_losses

def generate_loss_chart(log_path=None, output_image_path="qwen3_vl_loss_chart.png"):
    if not log_path:
        log_path = find_trainer_state_or_logs()

    steps, losses, eval_steps, eval_losses = extract_real_metrics(log_path)

    if not steps:
        raise RuntimeError(f"No real training metrics found in {log_path}")

    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=300)
    
    # Plot raw training loss
    ax.plot(steps, losses, color="#2563eb", linewidth=2.2, label="QLoRA Training Loss", alpha=0.85, zorder=3)
    
    # Moving average
    window = min(4, max(2, len(losses) // 10))
    if len(losses) >= window:
        smooth_loss = np.convolve(losses, np.ones(window)/window, mode="valid")
        ax.plot(steps[window-1:], smooth_loss, color="#dc2626", linewidth=2.4, linestyle="--", 
                label=f"Smoothed Trend (Window = {window})", zorder=4)

    # Validation loss dots
    if eval_losses:
        ax.scatter(eval_steps, eval_losses, color="#16a34a", s=80, zorder=5, 
                   label="Validation Loss", edgecolor="#0f172a", linewidth=0.8)

    # Epoch vertical lines
    max_step = max(steps)
    max_val = max(losses) if losses else 3.8
    for idx in [1, 2, 3]:
        ep_step = int(max_step * (idx / 3.0))
        if ep_step <= max_step:
            ax.axvline(x=ep_step, color="#94a3b8", linestyle=":", linewidth=1.5, alpha=0.8)
            ax.text(ep_step - 2.2, max_val * 0.94, f"Epoch {idx}", rotation=90, 
                    color="#475569", fontsize=9.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#e2e8f0", alpha=0.85))

    # Title and Labels - Clean publication title without "(Real Run)"
    ax.set_title("Training Loss Dynamics - Qwen3-VL-4B FMCG LoRA Adaptation", fontsize=14, fontweight="bold", pad=16)
    ax.set_xlabel("Optimization Steps (Effective Batch Size = 8)", fontsize=11, labelpad=10)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11, labelpad=10)
    
    # Grid and limits with comfortable headroom
    ax.grid(True, linestyle="--", alpha=0.45, color="#e2e8f0")
    ax.set_ylim(bottom=max(0, min(losses)*0.6 if losses else 0.2), top=max_val * 1.12)
    ax.set_xlim(left=-2, right=max_step + 4)

    # Legend in Upper Right
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=10, loc="upper right")
    
    # Info Box placed cleanly in the empty upper-middle area without any "REAL RUN" header
    final_loss_val = losses[-1] if losses else 0.6470
    eval_info = (
        f"\nBest Val Loss: {min(eval_losses):.4f}"
        f"\nFinal Val Loss: {eval_losses[-1]:.4f}"
        if eval_losses else ""
    )
    text_box = (
        f"Initial Loss: {losses[0]:.4f}\n"
        f"Final Loss: {final_loss_val:.4f}{eval_info}\n"
        f"LoRA Configuration: rank 16 / alpha 32\n"
        f"Target: FMCG Grounding"
    )
    ax.text(0.38, 0.66, text_box, transform=ax.transAxes, fontsize=9.5, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#94a3b8", alpha=0.95))

    plt.tight_layout()
    
    # Save to all required output paths
    paths_to_save = [
        output_image_path,
        "C:/HCMUT/Projects/HACKATHON/2nd_URA/Phase 3/qwen3_vl_loss_chart.png",
        "C:/HCMUT/Projects/HACKATHON/2nd_URA/Phase 3/qwen3_vl_4b_lora/artifacts/plots/qwen3_vl_loss_chart.png"
    ]
    for p in paths_to_save:
        out = Path(p)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out)
        print(f"Chart saved successfully at: {out.resolve()}")

if __name__ == "__main__":
    generate_loss_chart(output_image_path="C:/HCMUT/Projects/HACKATHON/2nd_URA/Phase 3/qwen3_vl_loss_chart.png")
