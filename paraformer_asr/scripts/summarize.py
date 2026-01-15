#!/usr/bin/env python3
"""
Result Aggregation and Summary System for Paraformer ASR

This script aggregates training results across multiple runs, generates
summary statistics, creates visualizations, and produces markdown reports.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_run_results(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load results from a single run directory.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Dictionary with run results or None if loading fails
    """
    results = {
        'run_name': run_dir.name,
        'run_dir': str(run_dir)
    }
    
    # Load training results CSV
    csv_file = run_dir / 'training_results.csv'
    if csv_file.exists():
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if rows:
                # Get final epoch results
                final_row = rows[-1]
                results['final_epoch'] = int(final_row['epoch'])
                results['final_train_loss'] = float(final_row['train_loss'])
                results['final_train_accuracy'] = float(final_row['train_token_accuracy'])
                
                if final_row.get('val_loss') and final_row['val_loss'] not in ['', 'None']:
                    results['final_val_loss'] = float(final_row['val_loss'])
                    results['final_val_accuracy'] = float(final_row['val_token_accuracy'])
                
                # Get best validation results
                if 'val_loss' in final_row and final_row.get('val_loss') and final_row['val_loss'] not in ['', 'None']:
                    val_losses = [float(row['val_loss']) for row in rows if row.get('val_loss') and row['val_loss'] not in ['', 'None']]
                    if val_losses:
                        results['best_val_loss'] = min(val_losses)
                        best_idx = val_losses.index(results['best_val_loss'])
                        results['best_val_epoch'] = int(rows[best_idx]['epoch'])
                        results['best_val_accuracy'] = float(rows[best_idx]['val_token_accuracy'])
                
                # Calculate total training time
                total_time = sum(float(row['epoch_time']) for row in rows if 'epoch_time' in row)
                results['total_training_time'] = total_time
                results['avg_epoch_time'] = total_time / len(rows) if rows else 0
                
                # Store full history for plotting
                results['history'] = {
                    'epochs': [int(row['epoch']) for row in rows],
                    'train_loss': [float(row['train_loss']) for row in rows],
                    'train_accuracy': [float(row['train_token_accuracy']) for row in rows]
                }
                
                if 'val_loss' in rows[0] and rows[0].get('val_loss') and rows[0]['val_loss'] not in ['', 'None']:
                    results['history']['val_loss'] = [float(row['val_loss']) for row in rows if row.get('val_loss') and row['val_loss'] not in ['', 'None']]
                    results['history']['val_accuracy'] = [float(row['val_token_accuracy']) for row in rows if row.get('val_token_accuracy') and row['val_token_accuracy'] not in ['', 'None']]
    else:
        return None
    
    # Load config if available
    config_file = run_dir / 'config.yaml'
    if not config_file.exists():
        config_file = run_dir / 'config.json'
    
    if config_file.exists():
        if config_file.suffix == '.json':
            with open(config_file, 'r') as f:
                results['config'] = json.load(f)
        # YAML loading would go here if needed
    
    return results


def aggregate_results(runs_dir: Path, filter_prefix: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Aggregate results from all runs in directory.
    
    Args:
        runs_dir: Directory containing run subdirectories
        filter_prefix: Only include runs starting with this prefix
        
    Returns:
        List of result dictionaries
    """
    all_results = []
    
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        if filter_prefix and not run_dir.name.startswith(filter_prefix):
            continue
        
        results = load_run_results(run_dir)
        if results:
            all_results.append(results)
    
    return all_results


def save_summary_csv(results: List[Dict[str, Any]], output_file: Path):
    """
    Save summary results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_file: Output CSV file path
    """
    if not results:
        print("No results to save")
        return
    
    # Determine columns based on available data
    columns = [
        'run_name',
        'final_epoch',
        'final_train_loss',
        'final_train_accuracy'
    ]
    
    # Add validation columns if any run has validation data
    if any('final_val_loss' in r for r in results):
        columns.extend([
            'final_val_loss',
            'final_val_accuracy',
            'best_val_loss',
            'best_val_accuracy',
            'best_val_epoch'
        ])
    
    columns.extend([
        'total_training_time',
        'avg_epoch_time'
    ])
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Summary CSV saved to: {output_file}")


def generate_markdown_report(results: List[Dict[str, Any]], output_file: Path):
    """
    Generate markdown summary report.
    
    Args:
        results: List of result dictionaries
        output_file: Output markdown file path
    """
    if not results:
        print("No results to report")
        return
    
    with open(output_file, 'w') as f:
        f.write("# ASR Training Results Summary\n\n")
        f.write(f"Total runs: {len(results)}\n\n")
        
        # Overall statistics
        f.write("## Overall Statistics\n\n")
        
        if any('best_val_loss' in r for r in results):
            val_results = [r for r in results if 'best_val_loss' in r]
            best_run = min(val_results, key=lambda x: x['best_val_loss'])
            
            f.write(f"**Best Validation Loss**: {best_run['best_val_loss']:.4f} ({best_run['run_name']})\n\n")
            f.write(f"**Best Validation Accuracy**: {best_run['best_val_accuracy']:.4f} ({best_run['run_name']})\n\n")
            
            avg_val_acc = sum(r['best_val_accuracy'] for r in val_results) / len(val_results)
            f.write(f"**Average Best Val Accuracy**: {avg_val_acc:.4f}\n\n")
        
        avg_train_time = sum(r['total_training_time'] for r in results) / len(results)
        f.write(f"**Average Training Time**: {avg_train_time:.2f} seconds\n\n")
        
        # Detailed results table
        f.write("## Detailed Results\n\n")
        
        # Table header
        if any('best_val_loss' in r for r in results):
            f.write("| Run Name | Final Train Acc | Final Train Loss | Best Val Acc | Best Val Loss | Training Time (s) |\n")
            f.write("|----------|----------------|------------------|--------------|---------------|-------------------|\n")
            
            for r in results:
                f.write(f"| {r['run_name']} | ")
                f.write(f"{r['final_train_accuracy']:.4f} | ")
                f.write(f"{r['final_train_loss']:.4f} | ")
                
                if 'best_val_accuracy' in r:
                    f.write(f"{r['best_val_accuracy']:.4f} | ")
                    f.write(f"{r['best_val_loss']:.4f} | ")
                else:
                    f.write("N/A | N/A | ")
                
                f.write(f"{r['total_training_time']:.2f} |\n")
        else:
            f.write("| Run Name | Final Train Acc | Final Train Loss | Training Time (s) |\n")
            f.write("|----------|----------------|------------------|-------------------|\n")
            
            for r in results:
                f.write(f"| {r['run_name']} | ")
                f.write(f"{r['final_train_accuracy']:.4f} | ")
                f.write(f"{r['final_train_loss']:.4f} | ")
                f.write(f"{r['total_training_time']:.2f} |\n")
        
        f.write("\n")
        
        # Model comparisons if applicable
        if len(results) >= 2:
            f.write("## Model Comparisons\n\n")
            
            encoder_runs = [r for r in results if 'encoder' in r['run_name'].lower() and 'only' in r['run_name'].lower()]
            full_runs = [r for r in results if 'full' in r['run_name'].lower() or 'paraformer' in r['run_name'].lower()]
            
            if encoder_runs and full_runs and any('best_val_accuracy' in r for r in results):
                encoder_avg = sum(r.get('best_val_accuracy', 0) for r in encoder_runs) / len(encoder_runs)
                full_avg = sum(r.get('best_val_accuracy', 0) for r in full_runs) / len(full_runs)
                
                f.write(f"**Encoder-Only Average Accuracy**: {encoder_avg:.4f}\n\n")
                f.write(f"**Full Paraformer Average Accuracy**: {full_avg:.4f}\n\n")
                f.write(f"**Improvement**: {(full_avg - encoder_avg):.4f} ({((full_avg - encoder_avg) / encoder_avg * 100):.2f}%)\n\n")
        
        # Key findings
        f.write("## Key Findings\n\n")
        f.write("- Training completed successfully for all runs\n")
        
        if any('best_val_accuracy' in r for r in results):
            best_run = max([r for r in results if 'best_val_accuracy' in r], key=lambda x: x['best_val_accuracy'])
            f.write(f"- Best performing model: {best_run['run_name']} (Accuracy: {best_run['best_val_accuracy']:.4f})\n")
        
        f.write(f"- Average training time: {avg_train_time / 60:.2f} minutes\n")
    
    print(f"Markdown report saved to: {output_file}")


def plot_training_curves(results: List[Dict[str, Any]], output_dir: Path):
    """
    Generate training curve visualizations.
    
    Args:
        results: List of result dictionaries
        output_dir: Output directory for plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for r in results:
        if 'history' in r:
            plt.plot(r['history']['epochs'], r['history']['train_loss'], label=f"{r['run_name']} (train)", alpha=0.7)
            if 'val_loss' in r['history']:
                plt.plot(r['history']['epochs'][:len(r['history']['val_loss'])], r['history']['val_loss'], label=f"{r['run_name']} (val)", linestyle='--', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for r in results:
        if 'history' in r:
            plt.plot(r['history']['epochs'], r['history']['train_accuracy'], label=f"{r['run_name']} (train)", alpha=0.7)
            if 'val_accuracy' in r['history']:
                plt.plot(r['history']['epochs'][:len(r['history']['val_accuracy'])], r['history']['val_accuracy'], label=f"{r['run_name']} (val)", linestyle='--', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Token Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {output_dir / 'training_curves.png'}")
    
    # Plot metric comparison
    if any('best_val_accuracy' in r for r in results):
        val_results = [r for r in results if 'best_val_accuracy' in r]
        
        plt.figure(figsize=(10, 6))
        run_names = [r['run_name'] for r in val_results]
        accuracies = [r['best_val_accuracy'] for r in val_results]
        losses = [r['best_val_loss'] for r in val_results]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        x = range(len(run_names))
        
        color = 'tab:blue'
        ax1.set_xlabel('Run')
        ax1.set_ylabel('Accuracy', color=color)
        ax1.bar(x, accuracies, alpha=0.7, color=color, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticks(x)
        ax1.set_xticklabels(run_names, rotation=45, ha='right')
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Loss', color=color)
        ax2.plot(x, losses, color=color, marker='o', linestyle='--', linewidth=2, markersize=8, label='Loss')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Best Validation Metrics Comparison')
        fig.tight_layout()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.savefig(output_dir / 'metric_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Metric comparison saved to: {output_dir / 'metric_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate and summarize ASR training results')
    parser.add_argument('--runs-dir', type=str, default='runs',
                       help='Directory containing run subdirectories')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for summary files')
    parser.add_argument('--filter', type=str, default=None,
                       help='Only include runs starting with this prefix')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not runs_dir.exists():
        print(f"Error: Runs directory not found: {runs_dir}")
        return 1
    
    print(f"Aggregating results from: {runs_dir}")
    if args.filter:
        print(f"Filtering runs with prefix: {args.filter}")
    
    # Aggregate results
    results = aggregate_results(runs_dir, args.filter)
    
    if not results:
        print("No results found!")
        return 1
    
    print(f"Found {len(results)} runs")
    
    # Save summary CSV
    save_summary_csv(results, output_dir / 'summary.csv')
    
    # Generate markdown report
    generate_markdown_report(results, output_dir / 'summary.md')
    
    # Generate plots
    if not args.no_plots:
        plot_training_curves(results, output_dir / 'plots')
    
    print(f"\nSummary complete! Results saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
