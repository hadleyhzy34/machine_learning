"""
Main script to run the Layer Normalization Ablation Study.

This script trains Vision Transformers with different normalization strategies
and compares their performance on CIFAR-10.

Variants:
- No-LN: No layer normalization (baseline)
- Post-LN: Original transformer (norm after attention/MLP)
- Pre-LN: Modern approach (norm before attention/MLP)
- RMSNorm: Root mean square normalization (used in LLaMA)
"""

import os
import sys
import json
import time
import torch
import numpy as np

from config import AblationConfig, ExperimentConfig, get_variant_description
from model import create_vit_model
from data import get_dataloaders
from train import train_model
from evaluate import (
    compute_convergence_metrics,
    detect_gradient_issues,
    generate_summary_report,
)
from visualize import generate_all_plots


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_experiment(
    config: ExperimentConfig,
    train_loader,
    test_loader,
    verbose: bool = True,
) -> dict:
    """
    Run a single experiment with the given configuration.

    Returns:
        Dictionary with training results and metrics
    """
    print(f"\n{'=' * 70}")
    print(f"Running experiment: {config.norm_type}")
    print(f"Description: {get_variant_description(config.norm_type)}")
    print(f"{'=' * 70}")

    # Create model
    model = create_vit_model(config)

    # Train model
    results = train_model(model, train_loader, test_loader, config, verbose=verbose)

    # Compute additional metrics
    results["convergence_metrics"] = compute_convergence_metrics(results["history"])
    results["gradient_analysis"] = detect_gradient_issues(
        results["history"]["gradient_norms"]
    )

    return results


def run_ablation_study(config: AblationConfig):
    """
    Run the complete ablation study across all variants.

    Returns:
        Dictionary with results for each variant
    """
    print("\n" + "=" * 70)
    print("LAYER NORMALIZATION ABLATION STUDY")
    print("=" * 70)
    print(f"\nDevice: {config.base_config.device}")
    print(f"Variants to test: {config.variants}")
    print(f"Epochs: {config.base_config.epochs}")
    print(f"Batch size: {config.base_config.batch_size}")
    print(f"Learning rate: {config.base_config.learning_rate}")

    # Set random seed
    set_seed(config.seed)

    # Load data once (shared across all experiments)
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader, class_names = get_dataloaders(
        batch_size=config.base_config.batch_size,
        device=config.base_config.device,
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Classes: {class_names}")

    # Create results directory
    os.makedirs(config.results_dir, exist_ok=True)

    # Run experiments for each variant
    all_results = {}

    for variant in config.variants:
        # Create config for this variant
        exp_config = ExperimentConfig(
            **{k: v for k, v in config.base_config.__dict__.items()},
        )
        exp_config.norm_type = variant

        # Run experiment
        results = run_single_experiment(
            exp_config, train_loader, test_loader, verbose=True
        )
        all_results[variant] = results

        # Save intermediate results
        save_results(all_results, config.results_dir, intermediate=True)

    # Generate final visualizations
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print("=" * 70)

    generate_all_plots(all_results, config.results_dir)

    # Generate and save summary report
    report = generate_summary_report(all_results, config)
    report_path = os.path.join(config.results_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nSaved summary report to {report_path}")

    # Save final results
    save_results(all_results, config.results_dir, intermediate=False)

    return all_results


def save_results(results: dict, output_dir: str, intermediate: bool = False):
    """Save results to JSON file."""
    # Convert numpy arrays and other non-serializable types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    serializable_results = convert_to_serializable(results)

    filename = "results_intermediate.json" if intermediate else "results_final.json"
    path = os.path.join(output_dir, filename)

    with open(path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Saved results to {path}")


def print_final_summary(results: dict):
    """Print a concise final summary to console."""
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Variant':<12} {'Final Acc':<12} {'Best Acc':<12} {'Time':<12}")
    print("-" * 48)

    for variant, r in results.items():
        print(
            f"{variant:<12} "
            f"{r['final_test_acc']:.2f}%      "
            f"{r['best_test_acc']:.2f}%      "
            f"{r['total_time']:.1f}s"
        )

    print("-" * 48)

    # Find best variant
    best_variant = max(results.items(), key=lambda x: x[1]["best_test_acc"])
    print(f"\nBest performing variant: {best_variant[0]} "
          f"({best_variant[1]['best_test_acc']:.2f}% accuracy)")


def main():
    """Main entry point."""
    # Create configuration
    config = AblationConfig(
        variants=["none", "post", "pre", "rms"],
        base_config=ExperimentConfig(
            patch_size=4,
            embed_dim=256,
            num_heads=8,
            num_layers=6,
            mlp_ratio=4,
            dropout=0.1,
            batch_size=128,
            epochs=20,
            learning_rate=3e-4,
            weight_decay=0.05,
        ),
        results_dir="results",
        seed=42,
    )

    # Run ablation study
    start_time = time.time()
    results = run_ablation_study(config)
    total_time = time.time() - start_time

    # Print summary
    print_final_summary(results)
    print(f"\nTotal ablation study time: {total_time / 60:.1f} minutes")
    print(f"\nResults saved to: {config.results_dir}/")


if __name__ == "__main__":
    main()
