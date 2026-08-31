"""
Unit Tests for Camera Manager Module (Multi-Camera & Smartphone Support)
"""

import unittest
from solidforge.core.camera_manager import CameraManager


class TestCameraManager(unittest.TestCase):
    def setUp(self):
        self.cam = CameraManager()

    def test_simulator_connection_and_frame_generation(self):
        connected = self.cam.connect_simulator()
        self.assertTrue(connected)
        self.assertTrue(self.cam.is_connected)
        self.assertEqual(self.cam.connection_mode, "SIMULATOR")

        frame = self.cam._generate_simulated_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape[0], 720)
        self.assertEqual(frame.shape[1], 1280)
        self.assertEqual(frame.shape[2], 3)

        self.cam.disconnect()
        self.assertFalse(self.cam.is_connected)

    def test_list_available_cameras(self):
        devices = self.cam.list_available_cameras(max_devices=2)
        self.assertIsInstance(devices, list)

    def test_smartphone_ip_connection_handling(self):
        # 無効なURL接続時の安全な拒否テスト
        success = self.cam.connect_smartphone_ip("http://0.0.0.0:9999/invalid")
        self.assertFalse(success)
        self.assertFalse(self.cam.is_connected)


if __name__ == "__main__":
    unittest.main()
