from utils import PostgresUtils
from utils import AzureUtils
from audio_analysis import AudioProsodyAnalysis

if __name__ == "__main__":
    azure_utils = AzureUtils(mount_dir="/mnt/data")

    # Check if running in Azure environment
    AZURE_RUN = azure_utils.detect_azure_run()

    if AZURE_RUN:
        print("Running in Azure environment")
        
        # Mount Azure storage if needed
        azure_utils.mount_dir_Azure()

        # DBFS paths
        input_folder_video_dbfs = f"{azure_utils.mount_dir}/audio/input/audio"
        output_folder_dbfs = f"{azure_utils.mount_dir}/audio/output"

        audio_analysis = AudioProsodyAnalysis(
            input_folder_video_dbfs,
            output_folder_dbfs,
            AZURE_RUN
        )
        path_rttm_file = audio_analysis.generate_rttm_from_xml(azure_utils.get_latest_xml, output_folder_dbfs)
        num_speakers = audio_analysis.count_speakers(audio_analysis.load_rttm(path_rttm_file))
        diarization = audio_analysis.diarization(azure_utils.get_latest_wav, num_speakers)
        speaker_segments, speaker_durations = audio_analysis.merge_and_analyze_speaker_segments(diarization)
        top5_speakers = audio_analysis.get_top_active_speakers(speaker_durations, 5)
        waveform_np, sample_rate = audio_analysis.load_waveform(azure_utils.get_latest_wav)
        prosody_data = audio_analysis.detailed_prosodic_analysis(waveform_np, sample_rate, speaker_segments, top5_speakers)
        audio_analysis.build_timeline_dataframe(prosody_data, top5_speakers)
        audio_analysis.export_timeline_to_csv(prosody_data, output_folder_dbfs)
        audio_analysis.export_intensity_distribution_to_csv(prosody_data, output_folder_dbfs)
        audio_analysis.export_pitch_distribution_to_csv(prosody_data, output_folder_dbfs)
        audio_analysis.export_prosodic_contours_to_csv(prosody_data, output_folder_dbfs)

    else:
        print("Not running in Azure environment, skipping Azure-specific operations.")
        pass