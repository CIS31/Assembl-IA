import os
import re
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torchaudio
import librosa

from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.metrics.diarization import DiarizationErrorRate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProsodyAnalysis:
    """
    Class for prosodic and diarization analysis of audio files.
    """

    def __init__(
        self,
        input_audio_dir: str,
        output_folder: str,
        azure_run: bool = False
    ):
        self.input_audio_dir = Path(input_audio_dir)
        self.input_retranscription_dir = Path(input_audio_dir)
    
        self.output_folder = Path(output_folder)
        self.azure_run = azure_run
        self.file_id = "parlementaire"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
        )
        self.pipeline.to(self.device)

    @staticmethod
    def load_rttm(rttm_path: str) -> Annotation:
        """
        Load an RTTM file and return a pyannote.core.Annotation.
        """
        annotation = Annotation()
        with open(rttm_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segment = Segment(start, start + duration)
                annotation[segment] = speaker
        return annotation

    @staticmethod
    def count_speakers(annotation: Annotation) -> int:
        """Count the number of unique speakers in a pyannote.core.Annotation."""
        return len(set(annotation.labels()))

    @staticmethod
    def load_waveform(audio_file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load an audio file and return the signal as a numpy array (mono) and sample rate.
        """
        waveform_torch, sample_rate = torchaudio.load(audio_file_path)
        waveform_np = waveform_torch[0].cpu().numpy() if waveform_torch.ndim > 1 else waveform_torch.squeeze().cpu().numpy()
        return waveform_np, sample_rate

    @staticmethod
    def get_top_active_speakers(speaker_durations: Dict[str, float], N: int = 3) -> List[str]:
        """Return the N speakers with the highest total speaking time."""
        return [label for label, _ in sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)[:N]]

    
    @staticmethod
    def build_timeline_dataframe(
        diarization: Annotation,
        speakers_labels: List[str]
    ) -> pd.DataFrame:
        """
        Build a DataFrame representing the timeline of the given speakers.

        Returns a DataFrame with columns: 'speaker_label', 'y_index', 'start', 'duration'.
        """
        speaker_to_y_index = {label: idx for idx, label in enumerate(speakers_labels)}
        plot_data = [
            {
                'speaker_label': label,
                'y_index': speaker_to_y_index[label],
                'start': segment.start,
                'duration': segment.duration
            }
            for segment, _, label in diarization.itertracks(yield_label=True)
            if label in speakers_labels
        ]
        return pd.DataFrame(plot_data)

    @staticmethod
    def export_pitch_distribution_to_csv(
        all_speaker_prosody_data: Dict[str, Any],
        top_active_speakers: List[str],
        csv_filename_pitch_dist: str
    ) -> None:
        """Export pitch distribution for each speaker to a CSV file."""
        pitch_data_for_export = [
            {'Speaker': speaker_label, 'Pitch_Hz': pitch_val}
            for speaker_label in top_active_speakers
            if speaker_label in all_speaker_prosody_data
            for pitch_val in all_speaker_prosody_data[speaker_label]['pitches_raw'][np.isfinite(all_speaker_prosody_data[speaker_label]['pitches_raw'])]
        ]
        df_pitch_distribution = pd.DataFrame(pitch_data_for_export)
        df_pitch_distribution.to_csv(csv_filename_pitch_dist, index=False)
        logger.info(f"Pitch distribution data exported to '{csv_filename_pitch_dist}'")

    @staticmethod
    def export_intensity_distribution_to_csv(
        all_speaker_prosody_data: Dict[str, Any],
        top_active_speakers: List[str],
        csv_filename_intensity_dist: str
    ) -> None:
        """Export intensity distribution for each speaker to a CSV file."""
        intensity_data_for_export = []
        for speaker_label in top_active_speakers:
            if speaker_label in all_speaker_prosody_data:
                raw_intensities = all_speaker_prosody_data[speaker_label]['intensities_raw']
                filtered_intensities = raw_intensities[np.isfinite(raw_intensities)]
                for intensity_val in filtered_intensities:
                    intensity_data_for_export.append({'Speaker': speaker_label, 'Intensity_dB': intensity_val})
            else:
                logger.warning(f"No prosodic data found for {speaker_label}, skipping export for this speaker.")

        df_intensity_distribution = pd.DataFrame(intensity_data_for_export)
        df_intensity_distribution.to_csv(csv_filename_intensity_dist, index=False)
        logger.info(f"Intensity distribution data exported to '{csv_filename_intensity_dist}'")

    @staticmethod
    def export_prosodic_contours_to_csv(
        all_speaker_prosody_data: Dict[str, Any],
        top_active_speakers: List[str],
        csv_filename_intensity_contour: str,
        csv_filename_pitch_contour: str
    ) -> None:
        """Export intensity and pitch contours for each speaker to CSV files."""
        # Intensity
        all_intensity_contour_data_for_export = [
            {'Speaker': speaker_label, 'Time_s': t, 'Intensity_dB': i_val}
            for speaker_label in top_active_speakers
            if speaker_label in all_speaker_prosody_data
            for t, i_val in zip(
                all_speaker_prosody_data[speaker_label]['intensities_times'][~np.isnan(all_speaker_prosody_data[speaker_label]['intensities_contour'])],
                all_speaker_prosody_data[speaker_label]['intensities_contour'][~np.isnan(all_speaker_prosody_data[speaker_label]['intensities_contour'])]
            )
        ]
        pd.DataFrame(all_intensity_contour_data_for_export).to_csv(csv_filename_intensity_contour, index=False)
        logger.info(f"Intensity contour data exported to '{csv_filename_intensity_contour}'")

        # Pitch
        all_pitch_contour_data_for_export = [
            {'Speaker': speaker_label, 'Time_s': t, 'Pitch_Hz': p}
            for speaker_label in top_active_speakers
            if speaker_label in all_speaker_prosody_data
            for t, p in zip(
                all_speaker_prosody_data[speaker_label]['pitches_times'][
                    (~np.isnan(all_speaker_prosody_data[speaker_label]['pitches_contour'])) &
                    (all_speaker_prosody_data[speaker_label]['pitches_contour'] > 0)
                ],
                all_speaker_prosody_data[speaker_label]['pitches_contour'][
                    (~np.isnan(all_speaker_prosody_data[speaker_label]['pitches_contour'])) &
                    (all_speaker_prosody_data[speaker_label]['pitches_contour'] > 0)
                ]
            )
        ]
        pd.DataFrame(all_pitch_contour_data_for_export).to_csv(csv_filename_pitch_contour, index=False)
        logger.info(f"Pitch contour data exported to '{csv_filename_pitch_contour}'")

    @staticmethod
    def export_timeline_to_csv(
        df_timeline: pd.DataFrame,
        csv_filename: str = "./output/timeline_speakers.csv"
    ) -> None:
        """Export the timeline of speakers to a CSV file."""
        df_timeline_export = df_timeline[['speaker_label', 'start', 'duration']]
        df_timeline_export.to_csv(csv_filename, index=False)
        logger.info(f"\n--- Données de la timeline exportées vers '{csv_filename}' ---")
        logger.info("Colonnes exportées : 'speaker_label', 'start', 'duration'")


    @staticmethod
    def export_diarization_to_rttm(
        diarization: Annotation,
        output_rttm_path: str,
        file_id: str
    ) -> None:
        """Export diarization results to RTTM format."""
        try:
            with open(output_rttm_path, 'w') as f:
                for segment, track_id, speaker_label in diarization.itertracks(yield_label=True):
                    line = (
                        f"SPEAKER {file_id} 1 {segment.start:.3f} {segment.duration:.3f} "
                        f"<NA> <NA> {speaker_label} <NA> <NA>\n"
                    )
                    f.write(line)
            logger.info(f"RTTM file generated at: {output_rttm_path}")
        except Exception as e:
            logger.error(f"Error generating RTTM file: {e}")

    @staticmethod
    def compute_der(reference_rttm_path: str, hypothesis_rttm_path: str) -> float:
        """Compute Diarization Error Rate (DER) between reference and hypothesis RTTM files."""
        reference = AudioProsodyAnalysis.load_rttm(reference_rttm_path)
        hypothesis = AudioProsodyAnalysis.load_rttm(hypothesis_rttm_path)
        der_metric = DiarizationErrorRate()
        der_value = der_metric(reference, hypothesis)
        logger.info(f"DER: {der_value:.2%}")
        return der_value

    def generate_rttm_from_xml(
        self,
        xml_file_path: str,
        output_dir: Optional[str] = None,
        file_id: Optional[str] = None,
        default_duration: float = 5.0
    ) -> Optional[str]:
        """Generate an RTTM file from an XML transcription file."""
        xml_file = Path(xml_file_path)
        if not xml_file.exists():
            logger.error(f"Error: XML file does not exist: {xml_file_path}")
            return None

        if file_id is None:
            file_id = xml_file.stem

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error(f"XML parsing error {xml_file_path}: {e}")
            return None

        ns = {'ns': 'http://schemas.assemblee-nationale.fr/referentiel'}
        rttm_lines = []
        orateur_id_map = {}

        for paragraphe in root.findall(".//ns:paragraphe", ns):
            texte_elem = paragraphe.find("ns:texte", ns)
            if texte_elem is not None and "stime" in texte_elem.attrib:
                try:
                    stime = float(texte_elem.attrib["stime"])
                except ValueError:
                    logger.warning(f"Warning: invalid stime '{texte_elem.attrib['stime']}'. Segment skipped.")
                    continue

                duration = default_duration
                orateur_elem = paragraphe.find(".//ns:orateur/ns:nom", ns)
                speaker = orateur_elem.text.strip() if orateur_elem is not None else "UNKNOWN"
                speaker_id = re.sub(r"[^\w]", "_", speaker)

                if speaker_id not in orateur_id_map:
                    orateur_id_map[speaker_id] = speaker

                rttm_line = (
                    f"SPEAKER {file_id} 1 {stime:.2f} {duration:.2f} <NA> <NA> {speaker_id} <NA> <NA>"
                )
                rttm_lines.append(rttm_line)

        output_rttm_path = (
            Path(output_dir) / f"{file_id}_reference_from_xml.rttm"
            if output_dir else Path(f"{file_id}_reference_from_xml.rttm")
        )
        output_rttm_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_rttm_path, "w", encoding="utf-8") as f:
            for line in rttm_lines:
                f.write(line + "\n")

        logger.info(f"RTTM file generated: {output_rttm_path} ({len(rttm_lines)} interventions).")
        return str(output_rttm_path)

    def diarization(self, audio_file_path: str, num_speakers: Optional[int] = None) -> Annotation:
        """Perform diarization on an audio file."""
        with ProgressHook() as hook:
            diarization = self.pipeline(audio_file_path, num_speakers=num_speakers, hook=hook)
        return diarization

    def merge_and_analyze_speaker_segments(
        self,
        diarization: Annotation,
        fusion_gap_threshold: float = 10
    ) -> Tuple[Dict[str, List[Segment]], Dict[str, float]]:
        """
        Merge speaker segments if the gap is below a threshold and compute total speaking time per speaker.
        """
        raw_speaker_segments = defaultdict(list)
        for segment, _, label in diarization.itertracks(yield_label=True):
            raw_speaker_segments[label].append(segment)

        speaker_segments = {}
        speaker_durations = {}

        for label, segments_list in raw_speaker_segments.items():
            if not segments_list:
                continue

            segments_list.sort(key=lambda s: s.start)
            merged_segments = []
            current_segment = segments_list[0]

            for next_segment in segments_list[1:]:
                gap = next_segment.start - current_segment.end
                if gap <= fusion_gap_threshold:
                    current_segment = Segment(current_segment.start, next_segment.end)
                else:
                    merged_segments.append(current_segment)
                    current_segment = next_segment
            merged_segments.append(current_segment)

            speaker_segments[label] = merged_segments
            speaker_durations[label] = sum(seg.duration for seg in merged_segments)

        return speaker_segments, speaker_durations

    def detailed_prosodic_analysis(
        self,
        waveform_np: np.ndarray,
        sample_rate: int,
        speaker_segments: Dict[str, List[Segment]],
        top_active_speakers: List[str],
        N_FFT: int = 1024,
        HOP_LENGTH: int = 512,
        MIN_SEGMENT_SAMPLES: Optional[int] = None,
        PITCH_MAGNITUDE_THRESHOLD: float = 0.01
    ) -> Dict[str, Any]:
        """
        Detailed prosodic analysis with pitch and intensity extraction for each speaker.
        """
        if MIN_SEGMENT_SAMPLES is None:
            MIN_SEGMENT_SAMPLES = N_FFT

        all_speaker_prosody_data = {}

        for speaker_label in top_active_speakers:
            speaker_pitches_raw = []
            speaker_intensities_raw = []

            speaker_pitches_contour = []
            speaker_pitches_times = []
            speaker_intensities_contour = []
            speaker_intensities_times = []

            for segment in speaker_segments[speaker_label]:
                start_sample = int(segment.start * sample_rate)
                end_sample = int(segment.end * sample_rate)
                segment_audio_np = waveform_np[start_sample:end_sample]

                if segment_audio_np.size < MIN_SEGMENT_SAMPLES:
                    continue

                pitches, magnitudes = librosa.core.piptrack(
                    y=segment_audio_np, sr=sample_rate,
                    fmin=75, fmax=500, n_fft=N_FFT, hop_length=HOP_LENGTH
                )
                f0_times_piptrack = librosa.times_like(pitches, sr=sample_rate, hop_length=HOP_LENGTH)

                reliable_pitch_frames_mask = np.zeros(pitches.shape[1], dtype=bool)
                pitch_values_segment_filtered = []

                for t_frame in range(pitches.shape[1]):
                    max_pitch_idx = magnitudes[:, t_frame].argmax()
                    pitch_val = pitches[max_pitch_idx, t_frame]
                    magnitude_val = magnitudes[max_pitch_idx, t_frame]
                    if pitch_val > 0 and magnitude_val > PITCH_MAGNITUDE_THRESHOLD:
                        pitch_values_segment_filtered.append(pitch_val)
                        reliable_pitch_frames_mask[t_frame] = True

                if pitch_values_segment_filtered:
                    speaker_pitches_raw.extend(pitch_values_segment_filtered)
                    speaker_pitches_contour.extend(pitch_values_segment_filtered)
                    speaker_pitches_times.extend(segment.start + f0_times_piptrack[reliable_pitch_frames_mask])

                rms = librosa.feature.rms(y=segment_audio_np, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
                intensity_db = librosa.amplitude_to_db(rms, ref=np.max)
                intensity_times_segment_raw = librosa.times_like(rms, sr=sample_rate, hop_length=HOP_LENGTH)

                min_len = min(len(reliable_pitch_frames_mask), len(intensity_db))
                reliable_intensity_values = intensity_db[:min_len][reliable_pitch_frames_mask[:min_len]]
                reliable_intensity_times = intensity_times_segment_raw[:min_len][reliable_pitch_frames_mask[:min_len]]

                if reliable_intensity_values.size > 0 and np.isfinite(reliable_intensity_values).any():
                    speaker_intensities_raw.extend(reliable_intensity_values[np.isfinite(reliable_intensity_values)])
                    speaker_intensities_contour.extend(reliable_intensity_values[np.isfinite(reliable_intensity_values)])
                    speaker_intensities_times.extend(segment.start + reliable_intensity_times[np.isfinite(reliable_intensity_values)])

            all_speaker_prosody_data[speaker_label] = {
                'pitches_raw': np.array(speaker_pitches_raw),
                'intensities_raw': np.array(speaker_intensities_raw),
                'pitches_contour': np.array(speaker_pitches_contour),
                'pitches_times': np.array(speaker_pitches_times),
                'intensities_contour': np.array(speaker_intensities_contour),
                'intensities_times': np.array(speaker_intensities_times),
            }

        return all_speaker_prosody_data
