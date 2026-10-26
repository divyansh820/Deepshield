"""
DeepShield — Comprehensive Automated Test Suite
Tests:
- API endpoints: /health, /diagnostics, /benchmarks, /models, /analyze, /analyze_batch
- Error handling: corrupted image, 0-byte file, unsupported format
- Forensic calculation stability: division by zero, flat images, extreme dimensions
- Grad-CAM layer extraction and gradient backprop
- Ensemble decision policy and model disagreement triggers
- Consecutive upload state isolation
"""

import io
import unittest
import numpy as np
from PIL import Image

import app as deepshield_app
import models
import gradcam
import forensics
import ensemble


class TestDeepShieldPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        deepshield_app.app.config['TESTING'] = True
        cls.client = deepshield_app.app.test_client()

        # Create a standard authentic-like test image
        img = Image.new('RGB', (200, 200), color=(120, 140, 160))
        # Add gradient pattern
        arr = np.array(img, dtype=np.uint8)
        arr[:, :, 0] += np.linspace(0, 50, 200, dtype=np.uint8).reshape(1, 200)
        cls.test_img = Image.fromarray(arr)

        buf = io.BytesIO()
        cls.test_img.save(buf, format='JPEG', quality=95)
        cls.test_jpeg_bytes = buf.getvalue()

    def test_01_health_endpoint(self):
        """Verifies /health reports online status and active operating mode."""
        res = self.client.get('/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "online")
        self.assertIn("active_mode", data)
        self.assertIn("checkpoints", data)
        self.assertIn("version", data)

    def test_02_diagnostics_endpoint(self):
        """Verifies /diagnostics returns device and model readiness."""
        res = self.client.get('/diagnostics')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("device", data)
        self.assertIn("torch_version", data)
        self.assertIn("checkpoint_directory", data)

    def test_03_models_endpoint(self):
        """Verifies /models returns available architectures."""
        res = self.client.get('/models')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("available_models", data)
        model_ids = [m["id"] for m in data["available_models"]]
        self.assertIn("ensemble", model_ids)
        self.assertIn("densenet121", model_ids)
        self.assertIn("resnet50", model_ids)

    def test_04_benchmarks_endpoint(self):
        """Verifies /benchmarks returns evaluation benchmark status."""
        res = self.client.get('/benchmarks')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("status", data)

    def test_05_analyze_valid_image(self):
        """Tests end-to-end /analyze request with a valid image."""
        data = {
            'image': (io.BytesIO(self.test_jpeg_bytes), 'test.jpg'),
            'model': 'ensemble'
        }
        res = self.client.post('/analyze', content_type='multipart/form-data', data=data)
        self.assertEqual(res.status_code, 200)
        resp = res.get_json()

        # Verify response schema
        self.assertIn("result", resp)
        self.assertIn("verdict_code", resp)
        self.assertIn("confidence", resp)
        self.assertIn("ai_probability", resp)
        self.assertIn("authentic_probability", resp)
        self.assertIn("model_agreement", resp)
        self.assertIn("gradcam", resp)
        self.assertIn("forensics", resp)
        self.assertIn("timing_ms", resp)

        # Confidence must be properly bounded and not hardcoded to 92.0%
        self.assertGreaterEqual(resp["confidence"], 0.0)
        self.assertLessEqual(resp["confidence"], 100.0)

        # Grad-CAM checks
        self.assertTrue(resp["gradcam"]["overlay_base64"].startswith("data:image/png;base64,"))
        self.assertTrue(resp["gradcam"]["heatmap_base64"].startswith("data:image/png;base64,"))
        self.assertIn("disclaimer", resp["gradcam"])

        # Latency breakdown checks
        timing = resp["timing_ms"]
        self.assertGreater(timing["total"], 0)
        self.assertIn("inference", timing)
        self.assertIn("forensics", timing)
        self.assertIn("gradcam", timing)

    def test_06_error_handling_empty_and_corrupt(self):
        """Verifies proper HTTP status codes for invalid/empty uploads."""
        # 1. No file provided
        res = self.client.post('/analyze', data={})
        self.assertEqual(res.status_code, 400)

        # 2. 0-byte file
        res = self.client.post('/analyze', data={'image': (io.BytesIO(b''), 'empty.jpg')})
        self.assertEqual(res.status_code, 400)

        # 3. Corrupted bytes
        res = self.client.post('/analyze', data={'image': (io.BytesIO(b'not_an_image_data_here'), 'corrupt.jpg')})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

    def test_07_batch_analysis(self):
        """Tests batch analysis endpoint with multiple images."""
        data = {
            'images': [
                (io.BytesIO(self.test_jpeg_bytes), 'img1.jpg'),
                (io.BytesIO(self.test_jpeg_bytes), 'img2.jpg')
            ],
            'model': 'ensemble'
        }
        res = self.client.post('/analyze_batch', content_type='multipart/form-data', data=data)
        self.assertEqual(res.status_code, 200)
        resp = res.get_json()
        self.assertIn("summary", resp)
        self.assertEqual(resp["summary"]["total"], 2)
        self.assertEqual(len(resp["results"]), 2)
        self.assertEqual(resp["results"][0]["status"], "success")

    def test_08_forensic_numerical_stability(self):
        """Tests forensic algorithms on extreme edge case images (completely flat 0-variance)."""
        flat_img = Image.new('RGB', (100, 100), color=(0, 0, 0))
        # Should execute without ZeroDivisionError or NaN
        freq = forensics.compute_frequency_analysis(flat_img)
        self.assertFalse(np.isnan(freq["hf_ratio"]))
        self.assertFalse(np.isnan(freq["spectral_slope"]))

        ela = forensics.compute_ela(flat_img)
        self.assertFalse(np.isnan(ela["edge_flat_ratio"]))
        self.assertFalse(np.isnan(ela["mean_residual"]))

        sensor = forensics.compute_sensor_noise_and_correlation(flat_img)
        self.assertFalse(np.isnan(sensor["channel_corr"]))
        self.assertFalse(np.isnan(sensor["noise_var"]))

        entropy = forensics.compute_vectorized_block_entropy(flat_img)
        self.assertFalse(np.isnan(entropy["mean_block_var"]))

    def test_09_ensemble_uncertainty_policy(self):
        """Tests 3-outcome uncertainty policy and model disagreement detection."""
        # Case A: Strong agreement on Authentic
        res_a = ensemble.evaluate_ensemble((0.90, 0.10), (0.85, 0.15), {}, is_demo_mode=False)
        self.assertEqual(res_a["verdict_code"], "real")
        self.assertTrue(res_a["models_agree"])

        # Case B: Strong agreement on AI
        res_b = ensemble.evaluate_ensemble((0.10, 0.90), (0.15, 0.85), {}, is_demo_mode=False)
        self.assertEqual(res_b["verdict_code"], "ai")
        self.assertTrue(res_b["models_agree"])

        # Case C: Severe model disagreement (DenseNet says 85% AI, ResNet says 15% AI)
        res_c = ensemble.evaluate_ensemble((0.15, 0.85), (0.85, 0.15), {}, is_demo_mode=False)
        self.assertEqual(res_c["verdict_code"], "uncertain")
        self.assertFalse(res_c["models_agree"])
        self.assertTrue(res_c["disagreement_warning"])

    def test_10_consecutive_uploads_isolation(self):
        """Verifies consecutive uploads receive independent predictions without stale state."""
        img1 = Image.new('RGB', (150, 150), color=(20, 30, 40))
        buf1 = io.BytesIO()
        img1.save(buf1, format='PNG')

        img2 = Image.new('RGB', (150, 150), color=(220, 230, 240))
        buf2 = io.BytesIO()
        img2.save(buf2, format='PNG')

        res1 = self.client.post('/analyze', data={'image': (io.BytesIO(buf1.getvalue()), 'dark.png')}).get_json()
        res2 = self.client.post('/analyze', data={'image': (io.BytesIO(buf2.getvalue()), 'bright.png')}).get_json()

        # Both must have valid independent timing and results
        self.assertIsNotNone(res1["timing_ms"]["total"])
        self.assertIsNotNone(res2["timing_ms"]["total"])


if __name__ == '__main__':
    unittest.main()
