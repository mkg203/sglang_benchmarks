"""
Plot SGLang Benchmark Results
Generates comparison plots from saved benchmark data.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict


def load_results(stats_file: str, results_file: str) -> Dict:
    """Load statistics and raw results."""
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return {'stats': stats, 'results': results}


def calculate_inter_token_latency(results: List[Dict]) -> Dict[str, float]:
    """Calculate inter-token latency percentiles from raw results."""
    itl_values = []
    for r in results:
        if r['output_tokens'] > 0:
            itl = r['decode_time'] / r['output_tokens']
            itl_values.append(itl)
    
    if not itl_values:
        return {'min': 0, 'p50': 0, 'p90': 0, 'p95': 0, 'p99': 0, 'max': 0}
    
    return {
        'min': np.min(itl_values),
        'p50': np.percentile(itl_values, 50),
        'p90': np.percentile(itl_values, 90),
        'p95': np.percentile(itl_values, 95),
        'p99': np.percentile(itl_values, 99),
        'max': np.max(itl_values)
    }


def plot_comparison(configs: List[Dict], output_file: str):
    """
    Generate comparison plot.
    
    configs: List of dicts with:
        - name: Configuration name
        - stats: Statistics dict
        - results: Raw results list
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SGLang Benchmark Results', fontsize=16, fontweight='bold')
    
    metrics = [
        ('time_to_first_token', 'Prefill Latency (TTFT)', axes[0, 0]),
        ('decode_latency', 'Decode Latency', axes[0, 1]),
        ('inter_token_latency', 'Inter-Token Latency', axes[1, 0]),
        ('total_latency', 'Total End-to-End Latency', axes[1, 1])
    ]
    
    percentile_keys = ['min', 'p50', 'p90', 'p95', 'p99', 'max']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c', '#9b59b6']
    
    x_positions = np.arange(len(configs))
    bar_width = 0.12
    
    for metric_key, title, ax in metrics:
        for i, pct in enumerate(percentile_keys):
            values = []
            
            for config in configs:
                if metric_key == 'inter_token_latency':
                    # Calculate from raw results
                    metric_data = calculate_inter_token_latency(config['results'])
                elif metric_key == 'decode_latency':
                    # Get from stats
                    metric_data = config['stats'].get('decode_latency', {})
                else:
                    # Get from stats
                    metric_data = config['stats'].get(metric_key, {})
                
                values.append(metric_data.get(pct, 0))
            
            positions = x_positions + (i - 2.5) * bar_width
            ax.bar(positions, values, bar_width, label=pct, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel('Latency (seconds)', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([c['name'] for c in configs], rotation=15, ha='right')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_file}")


def plot_throughput_comparison(configs: List[Dict], output_file: str):
    """Generate throughput comparison bar chart."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Throughput Comparison', fontsize=14, fontweight='bold')
    
    names = [c['name'] for c in configs]
    token_throughput = [c['stats']['throughput_tokens_per_sec'] for c in configs]
    request_throughput = [c['stats']['throughput_requests_per_sec'] for c in configs]
    
    x_pos = np.arange(len(names))
    
    ax1.bar(x_pos, token_throughput, color='#3498db', alpha=0.8)
    ax1.set_xlabel('Configuration', fontweight='bold')
    ax1.set_ylabel('Tokens/sec', fontweight='bold')
    ax1.set_title('Token Throughput')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.bar(x_pos, request_throughput, color='#e74c3c', alpha=0.8)
    ax2.set_xlabel('Configuration', fontweight='bold')
    ax2.set_ylabel('Requests/sec', fontweight='bold')
    ax2.set_title('Request Throughput')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Throughput plot saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot SGLang benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Single configuration
  python plot_results.py baseline_conc8
  
  # Compare multiple configurations
  python plot_results.py baseline_conc8 offload_conc8 --names "Baseline" "CPU Offload"
  
  # Multiple concurrency levels
  python plot_results.py baseline_conc1 baseline_conc4 baseline_conc8 \\
      --names "Conc 1" "Conc 4" "Conc 8"
        """
    )
    parser.add_argument("prefixes", nargs='+',
                       help="Output prefixes from benchmark runs (e.g., 'baseline_conc8')")
    parser.add_argument("--names", nargs='+',
                       help="Custom names for configurations (must match number of prefixes)")
    parser.add_argument("--output", default="benchmark_comparison.png",
                       help="Output plot filename")
    parser.add_argument("--throughput-plot", default="throughput_comparison.png",
                       help="Output throughput plot filename")
    
    args = parser.parse_args()
    
    # Validate
    if args.names and len(args.names) != len(args.prefixes):
        print(f"ERROR: Number of names ({len(args.names)}) must match number of prefixes ({len(args.prefixes)})")
        return
    
    # Load all configurations
    configs = []
    for i, prefix in enumerate(args.prefixes):
        stats_file = f"{prefix}_stats.json"
        results_file = f"{prefix}_results.json"
        
        if not Path(stats_file).exists() or not Path(results_file).exists():
            print(f"ERROR: Missing files for prefix '{prefix}'")
            print(f"  Expected: {stats_file} and {results_file}")
            continue
        
        print(f"Loading {prefix}...")
        data = load_results(stats_file, results_file)
        
        name = args.names[i] if args.names else prefix
        configs.append({
            'name': name,
            'stats': data['stats'],
            'results': data['results']
        })
    
    if not configs:
        print("ERROR: No valid configurations loaded")
        return
    
    print(f"\nGenerating plots for {len(configs)} configuration(s)...")
    
    # Generate plots
    plot_comparison(configs, args.output)
    plot_throughput_comparison(configs, args.throughput_plot)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
