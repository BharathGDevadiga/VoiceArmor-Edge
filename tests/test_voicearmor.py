import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from voicearmor_dsp import AudioDSPPipeline
from qnn_inference import VoiceArmorInferenceEngine
from arduino_alert import ArduinoHardwareBridge


class TestVoiceArmorEdge(unittest.TestCase):
    def setUp(self):
        self.dsp = AudioDSPPipeline()
        self.engine = VoiceArmorInferenceEngine(use_npu=False)
        self.arduino = ArduinoHardwareBridge(enabled=False)

    def test_dsp_spectrogram_dimensions(self):
        """Verify 250ms audio chunk (4000 samples @ 16kHz) produces [1, 1, 64, 96] tensor."""
        t = np.linspace(0, 0.25, 4000, endpoint=False)
        dummy_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        tensor = self.dsp.extract_log_mel_spectrogram(dummy_audio)

        self.assertEqual(tensor.shape, (1, 1, 64, 96))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertFalse(np.isnan(tensor).any())

    def test_dsp_short_buffer_handling(self):
        """Verify pipeline gracefully handles zero or undersized audio buffer."""
        short_audio = np.zeros(100, dtype=np.float32)
        tensor = self.dsp.extract_log_mel_spectrogram(short_audio)
        self.assertEqual(tensor.shape, (1, 1, 64, 96))

    def test_inference_engine_output_range(self):
        """Verify inference score is always a valid probability between 0.0 and 1.0."""
        dummy_tensor = np.random.randn(1, 1, 64, 96).astype(np.float32)
        prob = self.engine.predict_spoof_probability(dummy_tensor)

        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_arduino_bridge_simulation(self):
        """Verify Arduino hardware bridge commands execute without throwing exceptions."""
        try:
            self.arduino.trigger_threat_alert()
            self.arduino.clear_threat_alert()
            self.arduino.close()
        except Exception as e:
            self.fail(f"Arduino bridge raised unexpected exception: {e}")


if __name__ == "__main__":
    unittest.main()
