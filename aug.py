import os
import json
import numpy as np
from scipy.interpolate import interp1d

def load_initial_data(json_path):
    """Load initial data from the JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    positions = {}
    irs = {}
    base_path = os.path.dirname(json_path)

    for key, value in data.items():
        if key.startswith("ir_"):
            positions[key] = value["pos"]
            ir_path = os.path.join(base_path, f"{key}.npy")
            if os.path.exists(ir_path):
                irs[key] = np.load(ir_path)

    return data, positions, irs

def mbi_interpolation(pos1, pos2, ir1, ir2, steps):
    """Perform Modified Bilinear Interpolation (MBI)."""
    interpolated_positions = []
    interpolated_irs = []

    for i in range(1, steps):
        alpha = i / steps
        new_pos = {
            "x": (1 - alpha) * pos1["x"] + alpha * pos2["x"],
            "y": (1 - alpha) * pos1["y"] + alpha * pos2["y"],
            "z": (1 - alpha) * pos1["z"] + alpha * pos2["z"]
        }
        interpolated_positions.append(new_pos)

        new_ir = (1 - alpha) * ir1 + alpha * ir2
        interpolated_irs.append(new_ir)

    return interpolated_positions, interpolated_irs

def augment_data(json_path, output_folder, steps=2):
    """Augment data using MBI along x, y, and z axes."""
    os.makedirs(output_folder, exist_ok=True)

    # Load initial data
    data, positions, irs = load_initial_data(json_path)
    augmented_data = {}
    augmented_count = 0

    for key1, pos1 in positions.items():
        for key2, pos2 in positions.items():
            if key1 >= key2:
                continue

            ir1 = irs[key1]
            ir2 = irs[key2]

            # Check if positions are neighbors along any axis
            if (
                pos1["y"] == pos2["y"] and pos1["z"] == pos2["z"]
            ) or (
                pos1["x"] == pos2["x"] and pos1["z"] == pos2["z"]
            ) or (
                pos1["x"] == pos2["x"] and pos1["y"] == pos2["y"]
            ):
                interpolated_positions, interpolated_irs = mbi_interpolation(
                    pos1, pos2, ir1, ir2, steps
                )

                for i, (new_pos, new_ir) in enumerate(
                    zip(interpolated_positions, interpolated_irs)
                ):
                    new_key = f"ir_{augmented_count + len(positions)}"
                    augmented_data[new_key] = {
                        "time": None,  # Update if time information is needed
                        "pos": new_pos,
                    }
                    np.save(os.path.join(output_folder, f"{new_key}.npy"), new_ir)
                    augmented_count += 1

    # Merge augmented data with initial data
    data.update(augmented_data)

    # Save the updated JSON
    augmented_json_path = os.path.join(output_folder, "augmented_data.json")
    with open(augmented_json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Augmentation completed. Total IRs: {len(data)}")

