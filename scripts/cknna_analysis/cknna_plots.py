#%%
from pathlib import Path
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import os 
from dotenv import load_dotenv

# ======
# ONLY CHANGE THESE
BASELINE_TC = 202
REPA_TC = 203
# ======


load_dotenv()
#Set roots
REPO_ROOT = Path(os.environ["REPO_ROOT"])
DATA_ROOT = Path(os.environ["DATA_ROOT"])
RUNS_ROOT = Path(os.environ["RUNS_ROOT"])

CKNNA_ROOT = Path(f"{REPO_ROOT}/analysis/cknna/{BASELINE_TC}_{REPA_TC}")


#%%
def get_current_tc(path_name):
    match_obj = re.search(r"(?:^|_)tc(?P<tc>\d+)(?:_|$)", path_name) 
    if match_obj:
        return int(match_obj.group("tc"))
    else:
        return match_obj

def get_repa_config(tc):
    matches = []

    for config_path in RUNS_ROOT.rglob("config.json"):
        exp_name = config_path.parent.parent.name
        if get_current_tc(exp_name) != tc:
            continue

        with open(config_path) as file:
            config_file = json.load(file)

        config = config_file.get("config", config_file)
        if config.get("use_repa", False):
            matches.append(config)

    config = matches[0]
    student_layer = int(config["repa_student_layer"])
    student_layer = 12 if student_layer == -1 else student_layer
    repa_lambda = float(config["repa_lambda"])

    return student_layer, repa_lambda

#%%
ALIGNED_SL, REPA_LAMBDA = get_repa_config(REPA_TC)
ROOT = CKNNA_ROOT / f"tc{BASELINE_TC}_vs_tc{REPA_TC}_sl{ALIGNED_SL}_{REPA_LAMBDA}"

baseline = pd.read_csv(ROOT / f"single/tc{BASELINE_TC}/single_cknna.csv")
repa = pd.read_csv(ROOT / f"single/tc{REPA_TC}/single_cknna.csv")
trajectory = pd.read_csv(ROOT / f"comparison_tc{BASELINE_TC}vs.tc{REPA_TC}/trajectory.csv")

fig, ax = plt.subplots(figsize=(6, 3.8))

# ax.plot(baseline["student_layer"], baseline["cknna"], marker="o", color="#4C78A8", label="Pretrained ViT-S/16")
# ax.plot(repa["student_layer"], repa["cknna"], marker="o", color="#E45756", label="Pretrained ViT-S/16 + REPA-Seg (DINOv2-B/14 Teacher)")
ax.plot(baseline["student_layer"], baseline["cknna"], marker="o", color="#4C78A8", label="Scratch ViT-S/14")
ax.plot(repa["student_layer"], repa["cknna"], marker="o", color="#E45756", label="Scratch ViT-S/14 + REPA-Seg (DINOv2-B/14 Teacher)")
ax.axvline(ALIGNED_SL, linestyle="--", linewidth=1, color="#505963", label="Aligned student layer",)

ax.set_xlabel("Student layer")
ax.set_ylabel("CKNNA similarity to DINOv2-B/14")
#ax.set_xticks(range(13))
ax.set_axisbelow(True)
ax.grid(True, which="major", color="0.9", linewidth=0.8)
ax.legend( fontsize=8,loc="upper center", bbox_to_anchor=(0.5, -0.17),ncol=2)
plt.subplots_adjust(bottom=0.2)

ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax + 0.08 * (ymax - ymin))

fig.subplots_adjust(left=0.14,right=0.98,top=0.96,bottom=0.28)

OUTPUT_PATH = ROOT / f"cknna_by_layer_{BASELINE_TC}_{REPA_TC}.png"
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUTPUT_PATH)

#%%
fig, ax = plt.subplots(figsize=(6, 3.8))

# ax.plot(trajectory["epoch"], trajectory["baseline_cknna_score"], marker="o", color="#4C78A8", label="Pretrained ViT-S/16")
# ax.plot(trajectory["epoch"], trajectory["repa_cknna_score"], marker="o", color="#E45756", label="Pretrained ViT-S/16 + REPA-Seg (DINOv2-B/14 Teacher)")
ax.plot(trajectory["epoch"], trajectory["baseline_cknna_score"], marker="o", color="#4C78A8", label="Scratch ViT-S/14")
ax.plot(trajectory["epoch"], trajectory["repa_cknna_score"], marker="o", color="#E45756", label="Scratch ViT-S/14 + REPA-Seg (DINOv2-B/14 Teacher)")



ax.set_xlabel("Epoch")
ax.set_ylabel("CKNNA similarity to DINOv2-B/14")
ax.set_axisbelow(True)
ax.grid(True, which="major", color="0.9", linewidth=0.8)

ax.legend( fontsize=8,loc="upper center", bbox_to_anchor=(0.5, -0.17),ncol=1,)
plt.subplots_adjust(bottom=0.2)

ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax + 0.08 * (ymax - ymin))

fig.subplots_adjust(left=0.14,right=0.98,top=0.96,bottom=0.28)

OUTPUT_PATH = ROOT / f"cknna_trajectory_{BASELINE_TC}_{REPA_TC}.png"
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUTPUT_PATH)
# %%
