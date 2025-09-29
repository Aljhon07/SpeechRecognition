import torch
import os
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
train_loss = [2.0501407910406386, 0.7059756971928073, 0.6209609715087238, 0.6074306987826804, 0.5934000474021723, 0.5637342288967839, 0.5398919383564875, 0.5147075085162771, 0.47853658096851326, 0.4364800950730843, 0.39736474525203225, 0.3614951950169197, 0.3345181405299646, 0.3135654088064762, 0.3049]
dev_loss = [0.7740613136972699, 0.5925403102522805, 0.5708586648106575, 0.5550933914879957, 0.542438081687405, 0.5285603120213463, 0.5136609336449987, 0.49174588173627853, 0.46553919109560193, 0.4394769221544266, 0.4102686904370785, 0.396406352519989, 0.38343741425446104, 0.37704845349348726, 0.3760]

def load_checkpoint(checkpoint_path, device="cpu"):
    # This function is now unused but kept for context/completeness.
    # It will not be called in the main execution block.
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        return checkpoint["epoch_losses"]
    except FileNotFoundError:
        print(f"Warning: Checkpoint file not found at {checkpoint_path}")
        return {"train": [], "dev": []}

if __name__ == "__main__":
    # Use the initialized arrays directly
    train_losses = train_loss
    dev_losses = dev_loss

    print(f"Train Losses (first 5): {train_losses[:5]}")
    print(f"Validation Losses (first 5): {dev_losses[:5]}")

    # Ensure the lists are not empty before plotting
    if not train_losses or not dev_losses:
        print("Error: Loss lists are empty. Cannot create plot.")
    else:
        # Create the plot
        plt.figure(figsize=(8, 5))
        
        # Plotting Training Loss
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o", linestyle="-")
        
        # Plotting Validation Loss
        plt.plot(range(1, len(dev_losses) + 1), dev_losses, label="Validation Loss", marker="o", linestyle="-")

        # Add labels and title
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss Across Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Show the graph
        plt.show() # This displays the plot

        # Save as PNG
        plt.savefig("loss_plot.png", dpi=300)
        print("Plot saved as loss_plot.png")