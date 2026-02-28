import pandas as pd
import matplotlib.pyplot as plt

log_file = "checkpoints/.../progress_log_summary.csv"  # adjust path

data = pd.read_csv(log_file, sep="\t")

plt.figure()
plt.plot(data["train_loss"], label="Train Loss")
plt.plot(data["validation_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Convergence")
plt.grid()
plt.show()