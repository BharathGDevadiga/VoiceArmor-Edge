"""
VoiceArmor-Edge: Main Background Detection Service
Simulates or runs live continuous call monitoring with on-device NPU inference.
"""

import time
import argparse
import sys
import os
import numpy as np

# Ensure imports work when running from any directory (e.g. python src/main.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voicearmor_dsp import AudioDSPPipeline
from qnn_inference import VoiceArmorInferenceEngine
from arduino_alert import ArduinoHardwareBridge


def run_voicearmor(mode="simulated", threshold=0.85, use_npu=True):
    print("=" * 65)
    print(" VoiceArmor-Edge: Real-Time Voice Spoof Interceptor")
    print(" Target: Snapdragon-powered HP PCs + Arduino UNO Q")
    print("=" * 65)

    dsp = AudioDSPPipeline()
    engine = VoiceArmorInferenceEngine(use_npu=use_npu)
    arduino = ArduinoHardwareBridge()

    print(f"[STATUS] Background monitoring initiated. Threat threshold: {threshold}")
    print("[STATUS] Press Ctrl+C to stop.\n")

    alert_active = False

    try:
        sample_idx = 0
        while True:
            sample_idx += 1
            # In simulation mode, generate synthetic audio buffer with periodic injected spoofing
            t = np.linspace(0, 0.25, 4000, endpoint=False)
            if 15 <= (sample_idx % 30) <= 22:
                # Injected synthetic voice artifact: unnatural high-frequency carrier tones (vocoder buzz)
                audio_chunk = 0.6 * np.sin(2 * np.pi * 350 * t) + 0.4 * np.sin(2 * np.pi * 4200 * t)
                ground_truth = "SYNTHETIC/SPOOF"
            else:
                # Genuine vocal harmonics
                audio_chunk = 0.7 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
                ground_truth = "GENUINE VOICE"

            # 1. DSP Extraction
            t0 = time.perf_counter()
            spectrogram_tensor = dsp.extract_log_mel_spectrogram(audio_chunk)

            # 2. Hexagon NPU Inference
            t_inf_start = time.perf_counter()
            spoof_prob = engine.predict_spoof_probability(spectrogram_tensor)
            latency_ms = (time.perf_counter() - t_inf_start) * 1000

            # 3. Threat Classification
            if spoof_prob >= threshold:
                status_tag = "[THREAT DETECTED]"
                if not alert_active:
                    arduino.trigger_threat_alert()
                    alert_active = True
            else:
                status_tag = "[SAFE - VERIFIED]"
                if alert_active:
                    arduino.clear_threat_alert()
                    alert_active = False

            print(f"Frame #{sample_idx:03d} | Latency: {latency_ms:5.2f}ms | Spoof Score: {spoof_prob:4.2f} | {status_tag} (Input: {ground_truth})")
            time.sleep(0.25)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping VoiceArmor-Edge service...")
        arduino.clear_threat_alert()
        arduino.close()
        print("[INFO] Service shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoiceArmor-Edge Service")
    parser.add_argument("--device", choices=["npu", "cpu"], default="npu", help="Inference device")
    parser.add_argument("--threshold", type=float, default=0.85, help="Threat confidence threshold")
    args = parser.parse_args()

    run_voicearmor(threshold=args.threshold, use_npu=(args.device == "npu"))
