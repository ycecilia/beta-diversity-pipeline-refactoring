#!/usr/bin/env python3
"""
Simple but comprehensive benchmark comparing original beta.py vs refactored pipeline.
"""

import time
import subprocess
import psutil
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt

# Add paths
sys.path.append('/Users/ceciliayang/Desktop/report-homework-challenge/my_solution')
sys.path.append('/Users/ceciliayang/Desktop/report-homework-challenge')

def run_original_beta():
    """Run the original beta.py pipeline."""
    print("🔄 Running original beta.py...")
    
    # Set environment variables for original pipeline
    env = os.environ.copy()
    env['OTU_DATA_PATH'] = '/Users/ceciliayang/Desktop/report-homework-challenge/test_data/decontaminated_reads.csv'
    env['METADATA_PATH'] = '/Users/ceciliayang/Desktop/report-homework-challenge/test_data/sample_metadata.csv'
    env['CLUSTERING_METHOD'] = 'meanshift'
    
    # Measure execution
    start_time = time.time()
    initial_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
    
    try:
        result = subprocess.run([
            'python', '/Users/ceciliayang/Desktop/report-homework-challenge/original_code/beta.py'
        ], capture_output=True, text=True, env=env, 
           cwd='/Users/ceciliayang/Desktop/report-homework-challenge', timeout=300)
        
        end_time = time.time()
        final_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_delta = final_memory - initial_memory
        
        # Parse output for scientific metrics
        output = result.stdout + result.stderr
        scientific_results = {}
        
        # Extract metrics from output using regex or simple parsing
        import re
        
        # Look for PERMANOVA results with more flexible patterns
        f_stat_patterns = [
            r'F-statistic:\s*([0-9.]+)',
            r'F[- ]?stat[- ]?[:=]\s*([0-9.]+)',
            r'test statistic[- ]?[:=]\s*([0-9.]+)',
            r'Test statistic \(pseudo-F\):\s*([0-9.]+)',  # Match the exact format from original
            r'pseudo-F\):\s*([0-9.]+)',  # Shorter version
            r'PERMANOVA.*?F.*?([0-9.]+)',
        ]
        
        p_val_patterns = [
            r'p-value:\s*([0-9.]+)',
            r'p[- ]?val[- ]?[:=]\s*([0-9.]+)',
            r'PERMANOVA.*?p.*?([0-9.]+)',
            r'significance.*?([0-9.]+)',
        ]
        
        # Look for PCoA variance explained with more patterns
        pc1_patterns = [
            r'PC1.*?([0-9.]+)%',
            r'Principal Component 1.*?([0-9.]+)%',
            r'Axis 1.*?([0-9.]+)%',
            r'Component 1.*?([0-9.]+)%',
        ]
        
        pc2_patterns = [
            r'PC2.*?([0-9.]+)%',
            r'Principal Component 2.*?([0-9.]+)%',
            r'Axis 2.*?([0-9.]+)%',
            r'Component 2.*?([0-9.]+)%',
        ]
        
        # Look for distance matrix stats
        mean_dist_patterns = [
            r'[Mm]ean.*?distance.*?([0-9.]+)',
            r'Average.*?distance.*?([0-9.]+)',
            r'distance.*?mean.*?([0-9.]+)',
            r'distance.*?average.*?([0-9.]+)',
        ]
        
        # Try all patterns for each metric (handle trailing periods)
        for pattern in f_stat_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                # Remove any trailing period from the captured value
                value_str = match.group(1).rstrip('.')
                scientific_results['f_statistic'] = float(value_str)
                break
                
        for pattern in p_val_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                # Remove any trailing period from the captured value
                value_str = match.group(1).rstrip('.')
                scientific_results['p_value'] = float(value_str)
                break
                
        for pattern in pc1_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                scientific_results['pc1_variance'] = float(match.group(1))
                break
                
        for pattern in pc2_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                scientific_results['pc2_variance'] = float(match.group(1))
                break
                
        for pattern in mean_dist_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                scientific_results['mean_distance'] = float(match.group(1))
                break
        
        print(f"   ✅ Original completed in {execution_time:.2f}s")
        print(f"   📊 Return code: {result.returncode}")
        print(f"   💾 Memory change: {memory_delta:.1f}MB")
        
        return {
            'success': result.returncode == 0,
            'execution_time': execution_time,
            'memory_delta_mb': memory_delta,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
            'scientific_results': scientific_results
        }
        
    except subprocess.TimeoutExpired:
        print("   ❌ Original pipeline timed out after 5 minutes")
        return {'success': False, 'error': 'Timeout', 'execution_time': 300}
    except Exception as e:
        print(f"   ❌ Original pipeline failed: {e}")
        return {'success': False, 'error': str(e), 'execution_time': 0}

def run_refactored_beta(fast_mode=False):
    """Run the refactored beta diversity pipeline."""
    mode_str = "fast mode" if fast_mode else "normal mode"
    print(f"🚀 Running refactored pipeline ({mode_str})...")
    
    start_time = time.time()
    initial_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
    
    try:
        # Import and run refactored pipeline
        from beta_diversity_refactored.pipeline import BetaDiversityPipeline, PipelineInputs
        from beta_diversity_refactored.config import get_config
        
        # Configure pipeline
        config = get_config()
        if hasattr(config, 'analysis'):
            config.analysis.fast_mode = fast_mode
            
        pipeline = BetaDiversityPipeline(config)
        
        # Create inputs
        inputs = PipelineInputs(
            metadata_path=Path('/Users/ceciliayang/Desktop/report-homework-challenge/test_data/sample_metadata.csv'),
            abundance_path=Path('/Users/ceciliayang/Desktop/report-homework-challenge/test_data/decontaminated_reads.csv'),
            taxonomic_rank="species",
            environmental_param="site",
            beta_diversity_metric="jaccard",
            min_reads_per_sample=10,
            min_reads_per_taxon=5,
            permanova_permutations=999,
            enable_clustering=True,
            clustering_method="meanshift",
            output_prefix=f'benchmark_{"fast" if fast_mode else "normal"}',
            save_intermediate=False,
        )
        
        # Run pipeline
        outputs = pipeline.run(inputs)
        
        end_time = time.time()
        final_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_delta = final_memory - initial_memory
        
        print(f"   ✅ Refactored ({mode_str}) completed in {execution_time:.2f}s")
        print(f"   💾 Memory change: {memory_delta:.1f}MB")
        
        # Extract scientific results
        scientific_results = {}
        
        # Try to get cluster information
        if hasattr(outputs, 'cluster_info') and outputs.cluster_info:
            scientific_results['clusters'] = outputs.cluster_info.get('method_info', {}).get('n_clusters')
        
        # Try to read the saved JSON files for more accurate results
        output_prefix = f'benchmark_{"fast" if fast_mode else "normal"}'
        output_dir = Path('/Users/ceciliayang/Desktop/report-homework-challenge/my_solution/output')
        
        try:
            # Read PERMANOVA results from JSON file
            permanova_file = output_dir / f'{output_prefix}_permanova.json'
            if permanova_file.exists():
                import json
                with open(permanova_file, 'r') as f:
                    permanova_data = json.load(f)
                    scientific_results['f_statistic'] = permanova_data.get('test statistic')
                    scientific_results['p_value'] = permanova_data.get('p-value')
            
            # Read PCoA results from JSON file
            pcoa_file = output_dir / f'{output_prefix}_pcoa.json'
            if pcoa_file.exists():
                with open(pcoa_file, 'r') as f:
                    pcoa_data = json.load(f)
                    prop_explained = pcoa_data.get('proportion_explained', [])
                    if len(prop_explained) >= 2:
                        scientific_results['pc1_variance'] = prop_explained[0] * 100
                        scientific_results['pc2_variance'] = prop_explained[1] * 100
            
            # Read metadata for distance matrix stats
            metadata_file = output_dir / f'{output_prefix}_metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    # Calculate overall mean distance from dispersion data
                    dispersion = metadata.get('dispersion', {})
                    if dispersion:
                        all_distances = []
                        for site_data in dispersion.values():
                            mean_dist = site_data.get('mean_distance')
                            if mean_dist is not None:
                                all_distances.append(mean_dist)
                        if all_distances:
                            scientific_results['mean_distance'] = sum(all_distances) / len(all_distances)
            
            # Try to read distance matrix file for min/max if available
            distance_matrix_file = output_dir / f'{output_prefix}_distance_matrix.pkl.gz'
            if distance_matrix_file.exists():
                try:
                    import pickle
                    import gzip
                    import numpy as np
                    with gzip.open(distance_matrix_file, 'rb') as f:
                        distance_matrix = pickle.load(f)
                    
                    # Handle different matrix types
                    if hasattr(distance_matrix, 'data'):
                        matrix_data = distance_matrix.data
                    elif hasattr(distance_matrix, 'values'):
                        matrix_data = distance_matrix.values
                    else:
                        matrix_data = distance_matrix
                    
                    # Only use upper triangle (excluding diagonal) for distances
                    upper_triangle = matrix_data[np.triu_indices_from(matrix_data, k=1)]
                    if len(upper_triangle) > 0:
                        scientific_results['distance_min'] = float(np.min(upper_triangle))
                        scientific_results['distance_max'] = float(np.max(upper_triangle))
                        # Update mean distance with actual matrix calculation
                        scientific_results['mean_distance'] = float(np.mean(upper_triangle))
                except Exception as e:
                    print(f"   ⚠️  Could not read distance matrix: {e}")
                    
        except Exception as e:
            print(f"   ⚠️  Error reading JSON files: {e}")
            
        # Fallback to original method if JSON files didn't provide everything
        if hasattr(outputs, 'results') and outputs.results:
            # Try different ways to get PERMANOVA results
            permanova = None
            if hasattr(outputs.results, 'permanova_results'):
                permanova = outputs.results.permanova_results
            elif hasattr(outputs.results, 'statistical_results'):
                permanova = outputs.results.statistical_results.get('permanova')
            elif hasattr(outputs.results, 'analysis_results'):
                permanova = outputs.results.analysis_results.get('permanova_results')
                
            if permanova and 'f_statistic' not in scientific_results:
                # Try different key formats
                p_value = (permanova.get('p-value') or 
                          permanova.get('p_value') or 
                          permanova.get('pvalue'))
                f_stat = (permanova.get('test statistic') or 
                         permanova.get('f_statistic') or 
                         permanova.get('F-statistic') or
                         permanova.get('test_statistic'))
                         
                if p_value is not None and 'p_value' not in scientific_results:
                    scientific_results['p_value'] = float(p_value)
                if f_stat is not None and 'f_statistic' not in scientific_results:
                    scientific_results['f_statistic'] = float(f_stat)
                    
        # Print what we found for debugging
        print(f"   📊 Scientific metrics extracted: {list(scientific_results.keys())}")
        
        return {
            'success': True,
            'execution_time': execution_time,
            'memory_delta_mb': memory_delta,
            'scientific_results': scientific_results,
            'outputs': outputs
        }
        
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"   ❌ Refactored pipeline failed: {e}")
        return {
            'success': False, 
            'error': str(e), 
            'execution_time': execution_time
        }

def run_benchmark_iterations(n_iterations=3):
    """Run multiple benchmark iterations."""
    print(f"📊 Running {n_iterations} benchmark iterations...")
    
    results = {
        'original': [],
        'refactored': [],
        'refactored_fast': []
    }
    
    for i in range(n_iterations):
        print(f"\n🔄 Iteration {i+1}/{n_iterations}")
        
        # Run original
        orig_result = run_original_beta()
        results['original'].append(orig_result)
        
        # Run refactored normal
        refact_result = run_refactored_beta(fast_mode=False)
        results['refactored'].append(refact_result)
        
        # Run refactored fast
        refact_fast_result = run_refactored_beta(fast_mode=True)
        results['refactored_fast'].append(refact_fast_result)
        
        # Brief summary
        if orig_result['success'] and refact_result['success']:
            speedup = orig_result['execution_time'] / refact_result['execution_time']
            print(f"   🏃‍♂️ Speedup: {speedup:.2f}x")
    
    return results

def analyze_results(results):
    """Analyze benchmark results."""
    print("\n📈 Analyzing results...")
    
    analysis = {}
    
    for pipeline_name, pipeline_results in results.items():
        successful_runs = [r for r in pipeline_results if r['success']]
        
        if successful_runs:
            times = [r['execution_time'] for r in successful_runs]
            memories = [r.get('memory_delta_mb', 0) for r in successful_runs]
            
            analysis[pipeline_name] = {
                'execution_time': {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'samples': len(times)
                },
                'memory_usage': {
                    'mean': np.mean(memories),
                    'std': np.std(memories),
                },
                'success_rate': len(successful_runs) / len(pipeline_results) * 100
            }
            
            # Collect scientific results for consistency analysis
            scientific_data = []
            for run in successful_runs:
                if 'scientific_results' in run and run['scientific_results']:
                    scientific_data.append(run['scientific_results'])
            
            if scientific_data:
                analysis[pipeline_name]['scientific_results'] = scientific_data
        else:
            analysis[pipeline_name] = {
                'success_rate': 0,
                'error': 'All runs failed'
            }
    
    # Calculate performance improvements
    if 'original' in analysis and 'refactored' in analysis:
        # Check if both have execution_time data
        if ('execution_time' in analysis['original'] and 
            'execution_time' in analysis['refactored']):
            orig_time = analysis['original']['execution_time']['mean']
            refact_time = analysis['refactored']['execution_time']['mean']
            
            analysis['performance_comparison'] = {
                'speedup_factor': orig_time / refact_time if refact_time > 0 else 0,
                'time_improvement_pct': (orig_time - refact_time) / orig_time * 100 if orig_time > 0 else 0,
                'original_mean_time': orig_time,
                'refactored_mean_time': refact_time
            }
    
    # Add statistical consistency analysis
    analysis['statistical_consistency'] = analyze_statistical_consistency(analysis)
    
    return analysis

def analyze_statistical_consistency(analysis):
    """Compare statistical results between original and refactored pipelines."""
    print("🔬 Analyzing statistical consistency...")
    
    consistency_report = {
        'comparison_table': [],
        'status': 'unknown',
        'summary': {}
    }
    
    # Get scientific results from original and refactored runs
    original_results = analysis.get('original', {}).get('scientific_results', [])
    refactored_results = analysis.get('refactored', {}).get('scientific_results', [])
    
    if not original_results or not refactored_results:
        consistency_report['status'] = 'insufficient_data'
        consistency_report['summary'] = {
            'message': 'Insufficient scientific results data for comparison',
            'original_runs': len(original_results),
            'refactored_runs': len(refactored_results)
        }
        return consistency_report
    
    # Get the most recent results for comparison
    orig_result = original_results[-1] if original_results else {}
    refact_result = refactored_results[-1] if refactored_results else {}
    
    # Define metrics to compare
    metrics_to_compare = [
        ('f_statistic', 'PERMANOVA F-statistic', 2),
        ('p_value', 'PERMANOVA p-value', 3),
        ('pc1_variance', 'PCoA PC1 Variance', 2),
        ('pc2_variance', 'PCoA PC2 Variance', 2),
        ('mean_distance', 'Mean Distance', 4),
        ('distance_min', 'Distance Matrix Min', 1),
        ('distance_max', 'Distance Matrix Max', 1)
    ]
    
    comparison_results = []
    identical_count = 0
    total_comparisons = 0
    
    for metric_key, metric_name, decimal_places in metrics_to_compare:
        orig_val = orig_result.get(metric_key)
        refact_val = refact_result.get(metric_key)
        
        if orig_val is not None and refact_val is not None:
            total_comparisons += 1
            
            # Round values for comparison
            orig_rounded = round(float(orig_val), decimal_places)
            refact_rounded = round(float(refact_val), decimal_places)
            
            is_identical = abs(orig_rounded - refact_rounded) < (10 ** (-decimal_places))
            if is_identical:
                identical_count += 1
            
            # Format values for display
            if metric_key in ['pc1_variance', 'pc2_variance']:
                orig_display = f"{orig_rounded}%"
                refact_display = f"{refact_rounded}%"
            elif metric_key in ['distance_min', 'distance_max']:
                orig_display = f"[{orig_rounded}, 1.0]" if metric_key == 'distance_min' else f"[0.0, {orig_rounded}]"
                refact_display = f"[{refact_rounded}, 1.0]" if metric_key == 'distance_min' else f"[0.0, {refact_rounded}]"
            else:
                orig_display = str(orig_rounded)
                refact_display = str(refact_rounded)
            
            status = "✅ Identical" if is_identical else "❌ Different"
            
            comparison_results.append({
                'metric': f"**{metric_name}**",
                'original': orig_display,
                'refactored': refact_display,
                'status': status
            })
    
    consistency_report['comparison_table'] = comparison_results
    consistency_report['summary'] = {
        'identical_metrics': identical_count,
        'total_metrics': total_comparisons,
        'consistency_percentage': (identical_count / total_comparisons * 100) if total_comparisons > 0 else 0
    }
    
    if total_comparisons > 0:
        if identical_count == total_comparisons:
            consistency_report['status'] = 'perfect'
        elif identical_count >= total_comparisons * 0.8:
            consistency_report['status'] = 'good'
        else:
            consistency_report['status'] = 'poor'
    
    return consistency_report

def create_visualization(analysis):
    """Create performance comparison chart."""
    print("📊 Creating visualization...")
    
    # Extract data for plotting
    pipelines = []
    times = []
    errors = []
    
    pipeline_labels = {
        'original': 'Original',
        'refactored': 'Refactored', 
        'refactored_fast': 'Refactored (Fast)'
    }
    
    for pipeline_name, data in analysis.items():
        if pipeline_name != 'performance_comparison' and 'execution_time' in data:
            pipelines.append(pipeline_labels.get(pipeline_name, pipeline_name))
            times.append(data['execution_time']['mean'])
            errors.append(data['execution_time']['std'])
    
    if pipelines:
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        bars = plt.bar(pipelines, times, yerr=errors, capsize=5, 
                      color=['#ff7f0e', '#2ca02c', '#1f77b4'], alpha=0.8)
        
        plt.title('Beta Diversity Pipeline Performance Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Execution Time (seconds)', fontsize=12)
        plt.xticks(rotation=15)
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(errors)/2,
                    f'{time:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        results_dir = Path('/Users/ceciliayang/Desktop/report-homework-challenge/my_solution/benchmarks/results')
        results_dir.mkdir(exist_ok=True)
        plot_path = results_dir / 'simple_performance_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Chart saved to {plot_path}")
        return str(plot_path)
    
    return None

def save_results(results, analysis):
    """Save results to files."""
    print("💾 Saving results...")
    
    results_dir = Path('/Users/ceciliayang/Desktop/report-homework-challenge/my_solution/benchmarks/results')
    results_dir.mkdir(exist_ok=True)
    
    # Save raw results
    with open(results_dir / 'simple_benchmark_results.json', 'w') as f:
        # Convert any non-serializable objects
        clean_results = {}
        for pipeline, runs in results.items():
            clean_results[pipeline] = []
            for run in runs:
                clean_run = {k: v for k, v in run.items() if k != 'outputs'}  # Remove complex objects
                clean_results[pipeline].append(clean_run)
        json.dump(clean_results, f, indent=2, default=str)
    
    # Save analysis
    with open(results_dir / 'simple_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    # Create text summary
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("BETA DIVERSITY PIPELINE BENCHMARK RESULTS")
    summary_lines.append("=" * 60)
    
    if 'performance_comparison' in analysis:
        perf = analysis['performance_comparison']
        summary_lines.append(f"\n🚀 PERFORMANCE SUMMARY:")
        summary_lines.append(f"   • Original time: {perf['original_mean_time']:.3f}s")
        summary_lines.append(f"   • Refactored time: {perf['refactored_mean_time']:.3f}s")
        summary_lines.append(f"   • Speedup: {perf['speedup_factor']:.2f}x")
        summary_lines.append(f"   • Improvement: {perf['time_improvement_pct']:.1f}%")
    
    summary_lines.append(f"\n📊 DETAILED RESULTS:")
    for pipeline, data in analysis.items():
        if pipeline not in ['performance_comparison', 'statistical_consistency'] and 'execution_time' in data:
            summary_lines.append(f"\n{pipeline.upper()}:")
            summary_lines.append(f"   • Success rate: {data['success_rate']:.1f}%")
            summary_lines.append(f"   • Mean time: {data['execution_time']['mean']:.3f}s (±{data['execution_time']['std']:.3f}s)")
            summary_lines.append(f"   • Time range: {data['execution_time']['min']:.3f}s - {data['execution_time']['max']:.3f}s")
            summary_lines.append(f"   • Memory usage: {data['memory_usage']['mean']:.1f}MB (±{data['memory_usage']['std']:.1f}MB)")
    
    # Add statistical consistency section
    if 'statistical_consistency' in analysis:
        consistency = analysis['statistical_consistency']
        summary_lines.append(f"\n🔬 STATISTICAL CONSISTENCY:")
        
        if consistency['status'] == 'perfect':
            summary_lines.append(f"   ✅ Perfect consistency: {consistency['summary']['identical_metrics']}/{consistency['summary']['total_metrics']} metrics identical")
        elif consistency['status'] == 'good':
            summary_lines.append(f"   ✅ Good consistency: {consistency['summary']['identical_metrics']}/{consistency['summary']['total_metrics']} metrics identical ({consistency['summary']['consistency_percentage']:.1f}%)")
        elif consistency['status'] == 'poor':
            summary_lines.append(f"   ⚠️  Poor consistency: {consistency['summary']['identical_metrics']}/{consistency['summary']['total_metrics']} metrics identical ({consistency['summary']['consistency_percentage']:.1f}%)")
        else:
            summary_lines.append(f"   ❓ Insufficient data for consistency analysis")
        
        # Add detailed comparison table
        if consistency['comparison_table']:
            summary_lines.append(f"\n   METRIC COMPARISON:")
            summary_lines.append(f"   | Metric | Original | Refactored | Status |")
            summary_lines.append(f"   |--------|----------|------------|---------|")
            for row in consistency['comparison_table']:
                summary_lines.append(f"   | {row['metric']} | {row['original']} | {row['refactored']} | {row['status']} |")
    
    summary_lines.append("\n" + "=" * 60)
    
    with open(results_dir / 'simple_summary.txt', 'w') as f:
        f.write('\n'.join(summary_lines))
    
    # Print summary
    print('\n'.join(summary_lines))
    
    return results_dir

def main():
    """Main benchmark function."""
    print("🎯 SIMPLE BETA DIVERSITY PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Run benchmark
    results = run_benchmark_iterations(3)
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Create visualization
    plot_path = create_visualization(analysis)
    
    # Save results
    results_dir = save_results(results, analysis)
    
    print(f"\n🎉 Benchmark complete! Results saved to: {results_dir}")
    
    return results, analysis

if __name__ == "__main__":
    main()
