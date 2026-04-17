#!/usr/bin/env python3
"""
Comprehensive EEG Artifact Detection Module
==========================================

This module provides multiple artifact detection methods for EEG data:
1. Constant/flatline detection
2. High amplitude detection
3. Low variance detection
4. Step/jump detection
5. High frequency (muscle) detection

Each method can be used independently or combined through the
ComprehensiveArtifactDetector class.
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
import mne
import logging
from collections import defaultdict
from scipy.stats import gennorm

logger = logging.getLogger(__name__)


class ComprehensiveArtifactDetector:
    """
    Comprehensive artifact detection combining multiple methods.
    
    Parameters
    ----------
    min_constant_samples : int
        Minimum consecutive constant samples for flatline detection (default: 50)
    amplitude_z_threshold : float
        Z-score threshold for amplitude artifacts (default: 5.0)
    variance_percentile : float
        Percentile threshold for low variance detection (default: 5.0)
    step_threshold : float
        Threshold for step detection in standard deviations (default: 10.0)
    muscle_threshold : float
        Z-score threshold for muscle artifact detection (default: 4.0)
    """
    
    def __init__(
        self,
        flatline_min_samples: int = 50,
        flatline_grace_percent: float = 10.0,
        flatline_tolerance: float = None,
        variance_low_threshold_uV: float = 0.5,
        variance_high_threshold_uV: float = 195,
        variance_min_samples: int = 250,
        merge_overlap_ratio: float = 0.1,
        separate_annotations: bool = False
    ):
        
        self.flatline_min_samples = flatline_min_samples
        self.flatline_grace_percent = flatline_grace_percent
        self.flatline_tolerance = flatline_tolerance
        if self.flatline_tolerance is None:
            self.flatline_tolerance = np.finfo(float).eps
            
        self.variance_low_threshold_uV = variance_low_threshold_uV
        self.variance_high_threshold_uV = variance_high_threshold_uV
        self.variance_min_samples = variance_min_samples
        
        self.merge_overlap_ratio = merge_overlap_ratio
            
        self.separate_annotations = separate_annotations
        
    def detect_all_artifacts(self, raw: mne.io.Raw) -> Tuple[mne.Annotations, Dict[str, int]]:
        """
        Run all artifact detection methods and return annotations + summary.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data
            
        Returns
        -------
        annotations : mne.Annotations
            Detected artifact annotations
        summary : Dict[str, int]
            Count of each artifact type
        """
        all_artifacts: List[Tuple[float, float, str]] = []
        summary: Dict[str, int] = {}

        logger.debug("Detecting flatline artifacts...")
        flatline_artifacts = self._detect_flatline_artifacts(raw)
        all_artifacts.extend(flatline_artifacts)
        summary["Flatline"] = len(flatline_artifacts)

        logger.debug("Detecting variance artifacts...")
        var_artifacts = self._detect_variance_artifacts(raw)
        all_artifacts.extend(var_artifacts)
        summary["Variance"] = len(var_artifacts)
        if all_artifacts:
            merged = self._merge_overlapping_artifacts(all_artifacts)
            onsets, durations, descriptions = zip(*merged)
            annotations = mne.Annotations(onsets, durations, descriptions)
            return annotations, summary
        else:
            return mne.Annotations([], [], []), summary
    
    def _detect_flatline_artifacts(self, raw: mne.io.Raw) -> List[Tuple[float, float, str]]:
        """
        Detect flatline artifacts allowing up to `flatline_grace_percent` non-constant
        diffs, but segments must end on a diff marked as constant.

        Growth Strategy
        ---------------
        1. Start at a True in `is_almost_constant` and grow to the RIGHT, committing
           only when the end is True.
        2. When right growth cannot proceed without breaking grace constraint, grow
           LEFT while keeping the right end fixed (and True).
        3. Both the right end and the entire segment must satisfy the grace percentage
           constraint for inclusion.

        Returns
        -------
        list of (onset_sec, duration_sec, description)
            Detected flatline segments for this channel.
        """
        artifacts: List[Tuple[float, float, str]] = []
        data = raw.get_data()
        sample_rate_hz: float = float(raw.info["sfreq"])

        for channel_index, channel_name in enumerate(raw.ch_names):
            channel_values = data[channel_index]
            absolute_diffs = np.abs(np.diff(channel_values))

            is_almost_constant = absolute_diffs <= self.flatline_tolerance

            num_diffs = len(is_almost_constant)
            i = 0

            while i < num_diffs:
                if not is_almost_constant[i]:
                    i += 1
                    continue

                left = i
                right = i
                non_constant_count = 0
                j = i
                last_true_within_grace = i

                while j + 1 < num_diffs:
                    j += 1
                    if not is_almost_constant[j]:
                        non_constant_count += 1

                    current_length = j - left + 1
                    if (non_constant_count / current_length) <= self.flatline_grace_percent/100:
                        if is_almost_constant[j]:
                            last_true_within_grace = j
                            right = last_true_within_grace
                    else:
                        break

                k = left - 1
                while k >= 0:
                    added_non_const = 0 if is_almost_constant[k] else 1
                    new_length = right - k + 1
                    if (non_constant_count + added_non_const) / new_length <= self.flatline_grace_percent/100:
                        non_constant_count += added_non_const
                        left = k
                        k -= 1
                    else:
                        break

                segment_num_diffs = right - left + 1
                segment_num_samples = segment_num_diffs + 1  

                if segment_num_samples >= self.flatline_min_samples:
                    onset_sec = left / sample_rate_hz
                    duration_sec = segment_num_samples / sample_rate_hz
                    description = (
                        f"{channel_name}:Flatline" if self.separate_annotations else f"{channel_name}:Artifact"
                    )
                    artifacts.append((onset_sec, duration_sec, description))

                i = right + 1

        return artifacts


    def _detect_variance_artifacts(
        self,
        raw: mne.io.Raw,
    ) -> List[Tuple[float, float, str]]:
        """
        Detect per-channel maximal periods where the rolling standard deviation (window L)
        is either < low_threshold_uV or > high_threshold_uV (µV). Returns (onset_sec, duration_sec, label).

        Notes
        -----
        - Assumes `raw.get_data()` is in Volts (MNE default). Thresholds are given in µV.
        - Each returned interval is the union of consecutive windows whose std is consistently
        below the low threshold or above the high threshold.
        """
        artifacts: List[Tuple[float, float, str]] = []

        data = raw.get_data()
        sfreq = float(raw.info["sfreq"])
        window_length = int(max(1, self.variance_min_samples))
        low_threshold = float(self.variance_low_threshold_uV) * 1e-6
        high_threshold = float(self.variance_high_threshold_uV) * 1e-6

        for ch_idx, ch_name in enumerate(raw.ch_names):
            channel_data = data[ch_idx]
            num_samples = channel_data.size
            if num_samples < window_length:
                continue

            cumsum = np.cumsum(np.insert(channel_data, 0, 0.0))
            cumsum_sq = np.cumsum(np.insert(channel_data * channel_data, 0, 0.0))
            window_sum = cumsum[window_length:] - cumsum[:-window_length]
            window_sum_sq = cumsum_sq[window_length:] - cumsum_sq[:-window_length]
            window_mean = window_sum / window_length
            window_var = np.maximum(window_sum_sq / window_length - window_mean * window_mean, 0.0)
            window_std = np.sqrt(window_var)

            state = np.zeros_like(window_std, dtype=np.int8)
            state[window_std < low_threshold] = -1
            state[window_std > high_threshold] = +1
            if not np.any(state):
                continue

            padded_state = np.pad(state, (1, 1), constant_values=0)
            change_idx = np.flatnonzero(padded_state[1:] != padded_state[:-1])
            for i in range(len(change_idx) - 1):
                state_value = padded_state[change_idx[i] + 1]
                if state_value == 0:
                    continue
                window_start_idx = change_idx[i]
                window_end_idx = change_idx[i + 1]

                sample_start = window_start_idx
                sample_end = (window_end_idx - 1) + window_length
                duration_samples = sample_end - sample_start
                if duration_samples <= 0:
                    continue

                onset_sec = sample_start / sfreq
                duration_sec = duration_samples / sfreq
                if self.separate_annotations:
                    artifact_type = "LowVariance" if state_value < 0 else "HighVariance"
                else:
                    artifact_type = "Artifact"
                artifacts.append((onset_sec, duration_sec, f"{ch_name}:{artifact_type}"))

        return artifacts

    def _detect_gennorm_artifacts(
        self,
        raw,
        window_size: int = 1000,
        beta_threshold: float = 10.0,
        overlap: float = 0.0,
    ) -> List[Tuple[float, float, str]]:
        """
        Detect artifacts by fitting a generalized normal distribution (gennorm)
        on fixed-size windows of each channel.

        A window is marked as an artifact if the fitted shape parameter β exceeds
        the beta_threshold, indicating highly non-Gaussian signal distribution.

        Parameters
        ----------
        raw : mne.io.Raw
            The MNE Raw object.
        window_size : int
            Window length in samples (default: 1000).
        beta_threshold : float
            Threshold on β to flag an artifact (default: 10.0). Higher values
            indicate more non-Gaussian distributions.
        overlap : float
            Fractional overlap between consecutive windows in [0, 1). 
            0 = non-overlapping, 0.5 = 50% overlap.

        Returns
        -------
        artifacts : list of (onset_sec, duration_sec, description)
            Detected artifacts with onset in seconds, duration in seconds,
            and channel:ArtifactType description.
        """
        if gennorm is None:
            raise ImportError(
                "scipy is required for generalized normal fitting. "
                "Install scipy (e.g., `pip install scipy`)."
            )

        artifacts: List[Tuple[float, float, str]] = []
        data = raw.get_data()
        sfreq = float(raw.info["sfreq"])

        if not (0.0 <= overlap < 1.0):
            raise ValueError("overlap must be in [0, 1).")
        hop_size = max(1, int(round(window_size * (1.0 - overlap))))

        for ch_idx, ch_name in enumerate(raw.ch_names):
            channel_data = data[ch_idx]
            num_samples = channel_data.size
            if num_samples < window_size:
                continue

            window_start = 0
            while window_start + window_size <= num_samples:
                window_data = channel_data[window_start : window_start + window_size]

                if not np.isfinite(window_data).all():
                    window_data = window_data[np.isfinite(window_data)]
                if window_data.size < 10:
                    window_start += hop_size
                    continue
                if np.allclose(window_data, window_data[0]):
                    window_start += hop_size
                    continue

                try:
                    shape_param, location, scale = gennorm.fit(window_data)
                except Exception:
                    window_start += hop_size
                    continue

                if shape_param > beta_threshold:
                    onset_sec = window_start / sfreq
                    duration_sec = window_size / sfreq
                    artifact_label = f"{ch_name}:GenNorm" if getattr(self, "separate_annotations", False) else f"{ch_name}:Artifact"
                    artifacts.append((onset_sec, duration_sec, artifact_label))

                window_start += hop_size

        return artifacts

    def _merge_overlapping_artifacts(self, artifacts: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
        """
        Merge overlapping artifacts of the same type on the same channel.

        Uses a two-phase strategy:
        1. Collapse artifacts with 100 ms tolerance for obvious overlaps.
        2. Greedily merge nearby artifacts if relative growth is below merge_overlap_ratio.

        Returns
        -------
        list of (onset_sec, duration_sec, description)
            Merged artifacts sorted by onset time.
        """
        if not artifacts:
            return []

        groups = defaultdict(list)
        for onset, duration, desc in artifacts:
            groups[desc].append((onset, onset + duration))

        merged_all = []
        for desc, intervals in groups.items():
            intervals.sort()

            collapsed_intervals = []
            interval_start, interval_end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start <= interval_end + 0.1:
                    interval_end = max(interval_end, next_end)
                else:
                    collapsed_intervals.append((interval_start, interval_end))
                    interval_start, interval_end = next_start, next_end
            collapsed_intervals.append((interval_start, interval_end))

            def compute_growth_ratio(interval1, interval2):
                union_start = min(interval1[0], interval2[0])
                union_end = max(interval1[1], interval2[1])
                union_length = union_end - union_start
                total_length = (interval1[1] - interval1[0]) + (interval2[1] - interval2[0])
                return (union_length - total_length) / union_length if union_length > 0 else -1.0

            current_intervals = collapsed_intervals
            merge_occurred = True
            while merge_occurred and len(current_intervals) > 1:
                merge_occurred = False
                current_intervals.sort()
                for i in range(len(current_intervals)):
                    for j in range(i + 1, len(current_intervals)):
                        if compute_growth_ratio(current_intervals[i], current_intervals[j]) <= self.merge_overlap_ratio:
                            interval_a, interval_b = current_intervals[i], current_intervals[j]
                            current_intervals = current_intervals[:i] + current_intervals[i+1:j] + current_intervals[j+1:] + [(min(interval_a[0], interval_b[0]), max(interval_a[1], interval_b[1]))]
                            merge_occurred = True
                            break
                    if merge_occurred:
                        break

            merged_all.extend((start, end - start, desc) for start, end in current_intervals)

        return sorted(merged_all, key=lambda x: x[0])

    
    def get_artifact_summary(self, annotations: mne.Annotations) -> Dict[str, int]:
        """
        Get summary statistics of detected artifacts.

        Note:
            Since all annotation descriptions are now '<channel>:Artifact',
            this function will return a single 'Artifact' count unless you
            supply your own summary. Use the summary returned by
            `detect_all_artifacts` for per-type counts.
        """
        summary = {}
        
        for desc in annotations.description:
            if ':' in desc:
                _, art_type = desc.rsplit(':', 1)
                summary[art_type] = summary.get(art_type, 0) + 1
                
        return summary


def detect_comprehensive_artifacts(raw: mne.io.Raw, **kwargs) -> Tuple[mne.Annotations, Dict[str, int]]:
    """
    Convenience function to run comprehensive artifact detection.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    **kwargs : dict
        Additional parameters passed to ComprehensiveArtifactDetector
        
    Returns
    -------
    annotations : mne.Annotations
        Detected artifact annotations
    summary : Dict[str, int]
        Count of each artifact type
    """
    detector = ComprehensiveArtifactDetector(**kwargs)
    return detector.detect_all_artifacts(raw)