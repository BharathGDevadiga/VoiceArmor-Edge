"""
VoiceArmor-Edge: ONNX Runtime with Qualcomm QNN Execution Provider
Offloads spectrogram classification directly to the Snapdragon Hexagon NPU via QnnHtp.dll.
"""

import os
import numpy as np

class VoiceArmorInferenceEngine:
    def __init__(self, model_path="models/voicearmor_quantized.onnx", use_npu=True):
        self.model_path = model_path
        self.use_npu = use_npu
        self.session = self._initialize_session()

    def _initialize_session(self):
        try:
            import onnxruntime as ort
        except ImportError:
            print("[Warning] onnxruntime not installed. Running in mock simulation mode.")
            return None

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = []
        if self.use_npu:
            # Configure Qualcomm QNN Execution Provider for Hexagon Tensor Processor (HTP)
            qnn_options = {
                "backend_path": "QnnHtp.dll",
                "htp_performance_mode": "burst",
                "htp_graph_finalization_optimization_mode": "3"
            }
            providers.append(("QNNExecutionProvider", qnn_options))
            print("[INFO] Attempting execution on Qualcomm Hexagon NPU (QNN)...")

        # Fallback to CPU execution
        providers.append("CPUExecutionProvider")

        if os.path.exists(self.model_path):
            try:
                session = ort.InferenceSession(self.model_path, session_options, providers=providers)
                print(f"[INFO] Active Execution Provider: {session.get_providers()[0]}")
                return session
            except Exception as e:
                print(f"[Notice] Failed to load NPU session ({e}). Falling back to CPU.")
                return ort.InferenceSession(self.model_path, session_options, providers=["CPUExecutionProvider"])
        else:
            print(f"[INFO] Model weights file '{self.model_path}' not found on disk. Engine initialized in mock mode.")
            return None

    def predict_spoof_probability(self, spectrogram_tensor: np.ndarray) -> float:
        """
        Executes inference on 64-band Log-Mel Spectrogram tensor [1, 1, 64, 96].
        Returns:
            spoof_probability (float): Value between 0.0 (Genuine Voice) and 1.0 (Synthetic/Spoofed Voice).
        """
        if self.session is not None:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: spectrogram_tensor})
            # Softmax on binary logits
            logits = outputs[0][0]
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            return float(probs[1])  # Class 1: Synthetic Speech
        else:
            # Mock simulation for development testing
            # Simulates high-frequency harmonic analysis heuristic
            high_band_energy = np.mean(np.abs(spectrogram_tensor[0, 0, 48:64, :]))
            low_band_energy = np.mean(np.abs(spectrogram_tensor[0, 0, 0:16, :])) + 1e-6
            ratio = high_band_energy / low_band_energy
            prob = 1.0 / (1.0 + np.exp(-1.5 * (ratio - 0.7)))
            return float(np.clip(prob, 0.05, 0.98))
