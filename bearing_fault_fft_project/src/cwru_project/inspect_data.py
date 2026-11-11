import argparse
import os

import scipy.io


def main():
    parser = argparse.ArgumentParser(description="Inspect CWRU .mat files")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="folder containing the .mat files (default: data)",
    )
    args = parser.parse_args()

    files = {
        'Healthy': '97.mat',
        'Inner Race Fault': '105.mat',
        'Ball Fault': '118.mat',
        'Outer Race Fault': '130.mat',
    }

    print("--- Inspecting CWRU Dataset Keys ---\n")

    for label, filename in files.items():
        file_path = os.path.join(args.data_dir, filename)
        if os.path.exists(file_path):
            mat_data = scipy.io.loadmat(file_path)
            signal_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            print(f"[{label}] ({file_path}):")
            for key in signal_keys:
                print(f"  -> {key} (Shape: {mat_data[key].shape})")
            print("-" * 40)
        else:
            print(f"File missing: {file_path}")


if __name__ == "__main__":
    main()