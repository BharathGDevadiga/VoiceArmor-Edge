"""
VoiceArmor-Edge: Audio DSP Preprocessing Pipeline
Extracts 64-band Log-Mel Spectrogram features from incoming 16 kHz audio buffers.
"""

import numpy as np

SAMPLE_RATE = 16000
WINDOW_SIZE_MS = 25
HOP_SIZE_MS = 10
N_MELS = 64
BUFFER_DURATION_S = 0.25  # 250ms sliding window


class AudioDSPPipeline:
    def __init__(self, sample_rate=SAMPLE_RATE, n_mels=N_MELS):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.window_length = int(sample_rate * (WINDOW_SIZE_MS / 1000.0))
        self.hop_length = int(sample_rate * (HOP_SIZE_MS / 1000.0))
        self.n_fft = 512
        self.hanning_window = np.hanning(self.window_length)
        self.mel_filterbank = self._build_mel_filterbank()

    def _build_mel_filterbank(self):
        """Constructs a standard Mel filterbank matrix for 16 kHz audio."""
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (self.sample_rate / 2) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        filterbank = np.zeros((self.n_mels, int(self.n_fft // 2 + 1)))
        for m in range(1, self.n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                filterbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1] + 1e-8)
            for k in range(f_m, f_m_plus):
                filterbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m] + 1e-8)

        return filterbank

    def extract_log_mel_spectrogram(self, audio_buffer: np.ndarray) -> np.ndarray:
        """
        Converts 1D audio waveform buffer into 64-band Log-Mel Spectrogram tensor.
        Input: audio_buffer of shape [4000] (250ms at 16kHz)
        Output: Tensor of shape [1, 1, 64, 96] formatted for NPU inference.
        """
        if len(audio_buffer) < self.window_length:
            return np.zeros((1, 1, self.n_mels, 96), dtype=np.float32)

        # Frame the signal
        num_frames = 1 + int((len(audio_buffer) - self.window_length) / self.hop_length)
        frames = np.zeros((num_frames, self.window_length), dtype=np.float32)
        for i in range(num_frames):
            start = i * self.hop_length
            frames[i] = audio_buffer[start : start + self.window_length] * self.hanning_window

        # Compute Short-Time Fourier Transform (STFT) magnitude
        mag_frames = np.abs(np.fft.rfft(frames, n=self.n_fft))
        power_frames = (1.0 / self.n_fft) * (mag_frames ** 2)

        # Apply Mel Filterbank
        mel_energy = np.dot(power_frames, self.mel_filterbank.T)
        mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)
        log_mel = np.log(mel_energy).T  # Shape: [n_mels, num_frames]

        # Rescale / Pad / Crop to fixed 96 time-steps for model input
        target_time_steps = 96
        if log_mel.shape[1] < target_time_steps:
            pad_width = target_time_steps - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant")
        else:
            log_mel = log_mel[:, :target_time_steps]

        # Return standardized 4D Tensor [Batch, Channels, Height, Width]
        return np.expand_dims(np.expand_dims(log_mel.astype(np.float32), axis=0), axis=0)
