"""Entry point for Quantization Lab experiment pointers."""


def main() -> None:
    """Print the primary commands for the current sandbox experiments."""

    print("Quantization Lab is ready.")
    print("Run tests with:")
    print("  MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python -m pytest")
    print("Run Milestone 1 experiments with:")
    print("  MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/baseline_experiment.py")
    print("  MPLCONFIGDIR=/tmp/paroquant-mpl .venv/bin/python experiments/outlier_experiment.py")


if __name__ == "__main__":
    main()
