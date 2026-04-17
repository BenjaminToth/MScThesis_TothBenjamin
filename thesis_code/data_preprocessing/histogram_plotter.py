#!/usr/bin/env python3
"""
Generic histogram plotting utilities with specialized EEG preprocessing plots
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from collections import defaultdict
from scipy import stats
from scipy.optimize import minimize

import logging
logger = logging.getLogger(__name__)


class HistogramPlotter:
    """Generic histogram plotting utilities"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def save_figure(self, fig, filename: str):
        """Save figure in both PNG and PDF formats"""
        for fmt in ['pdf']:
            fig.savefig(self.output_dir / f'{filename}.{fmt}', 
                       dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def plot_bar_chart(self, 
                      data: Dict[Any, int], 
                      title: str,
                      xlabel: str,
                      ylabel: str,
                      filename: str,
                      show_counts: bool = True,
                      figsize: Tuple[int, int] = (10, 6)):
        """Create a bar chart with optional count labels"""
        if not data:
            logger.warning(f"No data for {filename}, skipping plot")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        keys = sorted(data.keys())
        values = [data[k] for k in keys]
        labels = [str(k) for k in keys]
        
        bars = ax.bar(labels, values, edgecolor='black', linewidth=1.2)
        
        if show_counts:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value}', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, filename)
    
    def plot_histogram(self,
                      data: List[float],
                      title: str,
                      xlabel: str,
                      ylabel: str,
                      filename: str,
                      bins: int = 30,
                      alpha: float = 0.7,
                      add_mean: bool = False,
                      add_median: bool = False,
                      convert_to_microvolts: bool = False,
                      figsize: Tuple[int, int] = (10, 6)):
        """
        Plot a histogram with optional statistical lines.
        
        Parameters
        ----------
        data : list[float]
            Data values to plot
        title : str
            Plot title
        xlabel : str
            X-axis label
        ylabel : str
            Y-axis label
        filename : str
            Output filename (without extension)
        bins : int
            Number of histogram bins
        alpha : float
            Transparency level (0-1)
        add_mean : bool
            Whether to plot a vertical line at the mean
        add_median : bool
            Whether to plot a vertical line at the median
        convert_to_microvolts : bool
            Convert data from volts to microvolts for display
        figsize : tuple
            Figure size (width, height)
        """
        if not data:
            logger.warning(f"No data for {filename}, skipping plot")
            return
        
        finite_data = [x for x in data if np.isfinite(x)]
        if not finite_data:
            logger.warning(f"No finite data for {filename}, skipping plot")
            return
        
        if convert_to_microvolts:
            finite_data = [x * 1e6 for x in finite_data]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        n, bins, patches = ax.hist(finite_data, bins=bins, edgecolor='black', linewidth=1.2, alpha=alpha)
        
        if add_mean:
            mean_value = np.mean(finite_data)
            ax.axvline(mean_value, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_value:.6f}')
        
        if add_median:
            median_value = np.median(finite_data)
            ax.axvline(median_value, color='orange', linestyle=':', linewidth=2,
                      label=f'Median: {median_value:.6f}')
        
        if add_mean or add_median:
            ax.legend()
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, filename)
    
    def plot_multi_histogram(self,
                           data_dict: Dict[str, List[float]],
                           title: str,
                           xlabel: str,
                           ylabel: str,
                           filename: str,
                           bins: int = 20,
                           alpha: float = 0.25,
                           figsize: Tuple[int, int] = (12, 6),
                           patterns: Optional[List[str]] = None):
        """
        Plot overlapping histograms for multiple categories, adding hatch
        patterns to distinguish the bars.

        Parameters
        ----------
        patterns : list[str] | None
            Hatch patterns to cycle through. Defaults to ['/', '\\\\', '|', '-'].
        """
        if patterns is None:
            patterns = ['/', '\\', '|', '-']
        
        filtered_data = {k: v for k, v in data_dict.items() if v}
        if not filtered_data:
            logger.warning(f"No data for {filename}, skipping plot")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        all_values = []
        for values in filtered_data.values():
            all_values.extend([v for v in values if np.isfinite(v) and v > 0])
        
        if not all_values:
            logger.warning(f"No positive finite values for {filename}, skipping plot")
            return
        
        bin_edges = np.linspace(0, max(all_values), bins)
        
        for i, (category, values) in enumerate(filtered_data.items()):
            clean_values = [v for v in values if np.isfinite(v) and v > 0]
            if clean_values:
                n, _, patches = ax.hist(
                    clean_values,
                    bins=bin_edges,
                    alpha=alpha,
                    label=category,
                    edgecolor='black',
                    linewidth=0.5,
                )
                hatch = patterns[i % len(patterns)]
                for p in patches:
                    p.set_hatch(hatch)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, filename)
    
    def plot_grouped_bar_chart(self,
                             data: Dict[str, Tuple[float, int]],
                             title: str,
                             xlabel: str,
                             ylabel: str,
                             filename: str,
                             value_label: str = 'n',
                             sort_by_value: bool = True,
                             figsize: Tuple[int, int] = (12, 8)):
        """
        Plot bar chart with values and counts displayed on bars.
        
        Parameters
        ----------
        data : dict[str, tuple[float, int]]
            Dictionary mapping labels to (value, count) tuples
        value_label : str
            Label to show for counts (e.g., 'n', 'count')
        sort_by_value : bool
            Sort bars by value (descending) if True, else alphabetically
        """
        if not data:
            logger.warning(f"No data for {filename}, skipping plot")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if sort_by_value:
            sorted_items = sorted(data.items(), key=lambda x: x[1][0], reverse=True)
        else:
            sorted_items = sorted(data.items())
        
        bar_labels = [item[0] for item in sorted_items]
        bar_values = [item[1][0] for item in sorted_items]
        bar_counts = [item[1][1] for item in sorted_items]
        
        x_positions = np.arange(len(bar_labels))
        bars = ax.bar(x_positions, bar_values, edgecolor='black', linewidth=1.2)
        
        for bar, count in zip(bars, bar_counts):
            bar_height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., bar_height,
                   f'{value_label}={count}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(bar_labels, rotation=45, ha='right')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, filename)
    
    def plot_amplitude_distribution(self,
                                   amplitudes: np.ndarray,
                                   filename: str,
                                   target_central_mass: float = 0.9999,
                                   num_bins: int = 1000,
                                   figsize: Tuple[int, int] = (12, 8)):
        """
        Plot amplitude distribution with fitted normal and generalized normal distributions.
        Performs all fitting calculations internally.
        
        Parameters
        ----------
        amplitudes : np.ndarray
            Flattened array of amplitude values in VOLTS
        filename : str
            Base filename for saving (without extension)
        target_central_mass : float
            Central mass for computing clipping thresholds
        num_bins : int
            Number of histogram bins
        """
        if not amplitudes.size:
            logger.warning(f"No amplitudes for {filename}, skipping plot")
            return None
        
        amplitudes_microvolts = amplitudes * 1e6
        
        fitted_normal_mu, fitted_normal_sigma = self._fit_normal_l1(amplitudes_microvolts, bins=num_bins)
        fitted_gennorm_beta, fitted_gennorm_loc, fitted_gennorm_scale = self._fit_gennorm_l1(amplitudes_microvolts, bins=num_bins)
        
        alpha = 1.0 - target_central_mass
        z_symmetric = stats.norm.ppf(1 - alpha / 2)
        half_width_empirical = z_symmetric * amplitudes_microvolts.std()
        half_width_normal = z_symmetric * fitted_normal_sigma
        half_width_gennorm = fitted_gennorm_scale * stats.gennorm.ppf(1 - alpha / 2, fitted_gennorm_beta)
        
        mean_amplitude = amplitudes_microvolts.mean()
        std_amplitude = amplitudes_microvolts.std()
        
        counts, bins = np.histogram(amplitudes_microvolts, bins=num_bins, density=True)
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.hist(amplitudes_microvolts, bins=bins, density=True, alpha=0.6, label='Data')
        
        x_axis = np.linspace(bins[0], bins[-1], 1000)
        
        pdf_normal = stats.norm.pdf(x_axis, loc=fitted_normal_mu, scale=fitted_normal_sigma)
        ax.plot(x_axis, pdf_normal, lw=2, label=f'L¹-fit Normal (σ={fitted_normal_sigma:.3g})', color='red')
        
        pdf_gennorm = stats.gennorm.pdf(x_axis, fitted_gennorm_beta, loc=fitted_gennorm_loc, scale=fitted_gennorm_scale)
        ax.plot(x_axis, pdf_gennorm, lw=2, label=f'L¹-fit GenNorm (β={fitted_gennorm_beta:.2f}, scale={fitted_gennorm_scale:.3g})', color='blue')
        
        ax.axvline(mean_amplitude, color='black', linestyle=':', label=f'μ ({mean_amplitude:.3g})')
        ax.axvline(fitted_normal_mu, color='red', linestyle=':', label=f'μ_norm ({fitted_normal_mu:.3g})')
        ax.axvline(fitted_gennorm_loc, color='blue', linestyle=':', label=f'μ_gn ({fitted_gennorm_loc:.3g})')
        
        ax.axvline(mean_amplitude + half_width_empirical, color='black', linestyle='--',
                   label=f'central {target_central_mass:.3%} emp σ-rule (+{half_width_empirical:.3g})')
        ax.axvline(mean_amplitude - half_width_empirical, color='black', linestyle='--')
        
        ax.axvline(fitted_normal_mu + half_width_normal, color='red', linestyle='--',
                   label=f'central {target_central_mass:.3%} normal (+{half_width_normal:.3g})', alpha=0.5)
        ax.axvline(fitted_normal_mu - half_width_normal, color='red', linestyle='--', alpha=0.5)
        
        ax.axvline(fitted_gennorm_loc + half_width_gennorm, color='blue', linestyle='--',
                   label=f'central {target_central_mass:.3%} gnorm (+{half_width_gennorm:.3g})', alpha=0.5)
        ax.axvline(fitted_gennorm_loc - half_width_gennorm, color='blue', linestyle='--', alpha=0.5)
        
        left_limit = 1.5*min(mean_amplitude - half_width_empirical, fitted_normal_mu - half_width_normal, fitted_gennorm_loc - half_width_gennorm)
        right_limit = 1.5*max(mean_amplitude + half_width_empirical, fitted_normal_mu + half_width_normal, fitted_gennorm_loc + half_width_gennorm)
        
        ax.set_xlim(max(amplitudes_microvolts.min(), left_limit), min(amplitudes_microvolts.max(), right_limit))
        ax.set_title(f"Amplitude Distribution (scaled) for {filename}")
        ax.set_xlabel("Amplitude (µV)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, f"{filename}_amplitude_histogram")
        
        return {
            'beta_gn': fitted_gennorm_beta,
            'loc_gn': fitted_gennorm_loc / 1e6,
            'scale_gn': fitted_gennorm_scale / 1e6,
            'clip_threshold': min(half_width_gennorm / 1e6, max(abs(amplitudes.min()), abs(amplitudes.max())))
        }
    
    def plot_venn_intersection_size_histogram(self,
                                            mask_durations: Dict[int, float],
                                            union_total: float,
                                            n_sources: int,
                                            figsize: Tuple[int, int] = (10, 6)):
        """
        Draw a cumulative stacked bar chart for Venn intersection sizes.
        
        Shows the percentage contribution of each intersection size (number of
        overlapping annotation sources) to the total annotated time.
        
        Parameters
        ----------
        mask_durations : Dict[int, float]
            Duration for each bitmask value (where bit position indicates source)
        union_total : float
            Total union duration across all annotations
        n_sources : int
            Number of annotation sources
        figsize : tuple
            Figure size (width, height)
        """
        if union_total <= 0 or not mask_durations:
            logger.warning("No data to plot Venn intersection size histogram.")
            return
        
        size_durations = defaultdict(float)
        for mask, duration in mask_durations.items():
            if mask == 0:
                continue
            num_sources_in_mask = self._popcount(mask)
            size_durations[num_sources_in_mask] += duration
        
        size_percentages = {k: (v / union_total) * 100.0 for k, v in size_durations.items()}
        for k in range(1, n_sources + 1):
            size_percentages.setdefault(k, 0.0)
        
        intersection_sizes = list(range(n_sources, 0, -1))
        percentage_values = [size_percentages[k] for k in intersection_sizes]
        
        x_indices = np.arange(len(intersection_sizes))
        colormap = plt.cm.get_cmap('tab10', n_sources)
        grey_background = "#dddddd"
        
        fig, ax = plt.subplots(figsize=figsize)
        cumulative_percentage = 0.0
        
        for idx, (intersection_size, contribution_percentage) in enumerate(zip(intersection_sizes, percentage_values)):
            if cumulative_percentage > 0:
                ax.bar(x_indices[idx], cumulative_percentage, color=grey_background, edgecolor='none')
            
            ax.bar(
                x_indices[idx],
                contribution_percentage,
                bottom=cumulative_percentage,
                edgecolor='black',
                linewidth=0.4,
                label=f'Size {intersection_size}' if idx == 0 else None
            )
            
            if contribution_percentage > 0:
                ax.text(
                    x_indices[idx],
                    cumulative_percentage + contribution_percentage / 2,
                    f"{contribution_percentage:.1f} %",
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='black'
                )
            
            cumulative_percentage += contribution_percentage
        
        ax.set_xticks(x_indices)
        ax.set_xticklabels([str(k) for k in intersection_sizes])
        ax.set_ylim(0, 100)
        ax.set_xlabel('Intersection size (number of sources)')
        ax.set_ylabel('Percentage of total annotated time (%)')
        ax.set_title('Cumulative Annotation Overlap by Intersection Size')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'annotation_venn_intersection_size_histogram')
    
    def _fit_normal_l1(self, data: np.ndarray, bins: int = 100) -> Tuple[float, float]:
        """
        Fit a normal distribution to data by minimizing L¹ (absolute) error.
        
        Uses robust initial estimates based on median and MAD (median absolute
        deviation), then optimizes parameters with Nelder-Mead simplex method.
        
        Returns
        -------
        mu : float
            Estimated mean
        sigma : float
            Estimated standard deviation
        """
        counts, bin_edges = np.histogram(data, bins=bins, density=False)
        bin_widths = np.diff(bin_edges)
        probability_density = counts / counts.sum() / bin_widths
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        def objective_function(parameters: np.ndarray) -> float:
            mean, log_sigma = parameters
            sigma = np.exp(log_sigma)
            predicted_pdf = stats.norm.pdf(bin_centers, loc=mean, scale=sigma)
            predicted_pdf = predicted_pdf / (np.sum(predicted_pdf * bin_widths))
            return np.sum(np.abs(probability_density - predicted_pdf) * bin_widths)
        
        median_estimate = np.median(data)
        mad_estimate = np.median(np.abs(data - median_estimate))
        sigma_estimate = mad_estimate * 1.4826 if mad_estimate > 0 else np.std(data)
        result = minimize(objective_function, x0=[median_estimate, np.log(sigma_estimate)], method='Nelder-Mead')
        fitted_mean, fitted_log_sigma = result.x
        return fitted_mean, float(np.exp(fitted_log_sigma))
    
    def _fit_gennorm_l1(self, data: np.ndarray, bins: int = 100) -> Tuple[float, float, float]:
        """
        Fit a generalized normal distribution to data by minimizing L¹ error.
        
        Enforces shape parameter β ≥ 1 for numerical stability. Uses robust
        initial estimates based on median and MAD, optimizing in log-space.
        
        Returns
        -------
        beta : float
            Shape parameter (β >= 1)
        location : float
            Location/mean parameter
        scale : float
            Scale parameter
        """
        counts, bin_edges = np.histogram(data, bins=bins, density=False)
        bin_widths = np.diff(bin_edges)
        probability_density = counts / counts.sum() / bin_widths
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        def objective_function(parameters: np.ndarray) -> float:
            log_beta, location, log_scale = parameters
            shape_param = 1.0 + np.exp(log_beta)
            scale = np.exp(log_scale)
            predicted_pdf = stats.gennorm.pdf(bin_centers, shape_param, loc=location, scale=scale)
            predicted_pdf = predicted_pdf / (np.sum(predicted_pdf * bin_widths))
            return np.sum(np.abs(probability_density - predicted_pdf) * bin_widths)
        
        median_estimate = np.median(data)
        mad_estimate = np.median(np.abs(data - median_estimate))
        scale_estimate = mad_estimate * 1.4826 if mad_estimate > 0 else np.std(data)
        log_beta_estimate = np.log(2.0)
        result = minimize(objective_function, x0=[log_beta_estimate, median_estimate, np.log(scale_estimate)], method='Nelder-Mead')
        fitted_log_beta, fitted_location, fitted_log_scale = result.x
        fitted_shape_param = 1.0 + np.exp(fitted_log_beta)
        fitted_scale = np.exp(fitted_log_scale)
        return fitted_shape_param, fitted_location, fitted_scale
    
    def _popcount(self, mask: int) -> int:
        """
        Count the number of 1-bits in a bitmask integer.
        
        Portable implementation compatible with Python < 3.8 (which lacks
        the built-in popcount method).
        """
        return bin(mask).count("1")
