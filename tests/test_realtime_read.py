import struct
import time
import unittest

from phyling.ble.base_device import (
    ACC_FACTOR,
    GYRO_FACTOR,
    NOTIF_DIFF_OFFSET,
    _make_col_spec,
)
from phyling.ble.nanophyling import NANO_DEF_CONFIG, NanoPhyling


def _build_device() -> NanoPhyling:
    """Return a NanoPhyling instance with config and df ready (no BLE needed)."""
    device = NanoPhyling(ble_name="NanoPhyling_Test", address="00:00:00:00:00:00")
    device.config = {**NANO_DEF_CONFIG}
    device._col_specs = [_make_col_spec(col, "B") for col in device.config["data"]]
    device._oneDataSize = sum(s["size"] for s in device._col_specs)
    device._init_df_if_needed()
    device.startPCtime = time.time()
    device.startBLETime = 1.0  # will be overwritten on first notification
    return device


def _build_notification(ble_time_us: int, spacing_us: int, samples: list[list[int]]) -> bytes:
    """
    Build a raw BLE notification packet.

    :param ble_time_us: Current BLE time in microseconds
    :param spacing_us: Sample spacing in microseconds
    :param samples: List of sample rows; each row is a list of int16 raw sensor values
                    in the order [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    """
    header = struct.pack("<QH", ble_time_us, spacing_us)
    body = b""
    for sample in samples:
        for val in sample:
            body += struct.pack("<h", val)  # int16 little-endian
    return header + body


class TestOnData(unittest.TestCase):

    def test_callback_is_called_once_per_notification(self):
        device = _build_device()
        calls = []
        device.on_data(lambda df: calls.append(df))

        packet = _build_notification(
            ble_time_us=1_000_000,
            spacing_us=5_000,
            samples=[[100, 200, 300, 10, 20, 30]],
        )
        device._notification_handler(None, packet)

        self.assertEqual(len(calls), 1)

    def test_callback_receives_correct_columns(self):
        device = _build_device()
        received = []
        device.on_data(lambda df: received.append(df))

        packet = _build_notification(
            ble_time_us=1_000_000,
            spacing_us=5_000,
            samples=[[100, 200, 300, 10, 20, 30]],
        )
        device._notification_handler(None, packet)

        df = received[0]
        expected_cols = {"T", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"}
        self.assertEqual(set(df.columns), expected_cols)

    def test_callback_receives_one_row_per_sample(self):
        device = _build_device()
        received = []
        device.on_data(lambda df: received.append(df))

        packet = _build_notification(
            ble_time_us=1_000_000,
            spacing_us=5_000,
            samples=[[100, 200, 300, 10, 20, 30], [110, 210, 310, 11, 21, 31]],
        )
        device._notification_handler(None, packet)

        self.assertEqual(len(received[0]), 2)

    def test_callback_applies_calibration(self):
        device = _build_device()
        device.calibration = {"acc_x": {"coef": 2.0, "offset": 5.0}}
        received = []
        device.on_data(lambda df: received.append(df))

        raw_acc_x = 100
        packet = _build_notification(
            ble_time_us=1_000_000,
            spacing_us=5_000,
            samples=[[raw_acc_x, 0, 0, 0, 0, 0]],
        )
        device._notification_handler(None, packet)

        df = received[0]
        # Expected: coef * (raw_value * ACC_FACTOR + offset) = 2.0 * (100 * ACC_FACTOR + 5.0)
        expected = 2.0 * (raw_acc_x * ACC_FACTOR + 5.0)
        self.assertAlmostEqual(df["acc_x"].iloc[0], expected, places=6)

    def test_callback_sensor_values_use_scale_factor(self):
        device = _build_device()
        received = []
        device.on_data(lambda df: received.append(df))

        raw_acc_x = 1000
        raw_gyro_z = 500
        packet = _build_notification(
            ble_time_us=1_000_000,
            spacing_us=5_000,
            samples=[[raw_acc_x, 0, 0, 0, 0, raw_gyro_z]],
        )
        device._notification_handler(None, packet)

        df = received[0]
        self.assertAlmostEqual(df["acc_x"].iloc[0], raw_acc_x * ACC_FACTOR, places=6)
        self.assertAlmostEqual(df["gyro_z"].iloc[0], raw_gyro_z * GYRO_FACTOR, places=6)

    def test_callback_t_is_non_negative(self):
        device = _build_device()
        received = []
        device.on_data(lambda df: received.append(df))

        packet = _build_notification(
            ble_time_us=1_000_000,
            spacing_us=5_000,
            samples=[[0, 0, 0, 0, 0, 0]],
        )
        device._notification_handler(None, packet)

        self.assertGreaterEqual(received[0]["T"].iloc[0], 0.0)

    def test_no_callback_does_not_raise(self):
        device = _build_device()
        packet = _build_notification(
            ble_time_us=1_000_000,
            spacing_us=5_000,
            samples=[[100, 200, 300, 10, 20, 30]],
        )
        # Should not raise even with no callback registered
        device._notification_handler(None, packet)

    def test_callback_accumulates_across_multiple_notifications(self):
        device = _build_device()
        all_dfs = []
        device.on_data(lambda df: all_dfs.append(df))

        for i in range(3):
            packet = _build_notification(
                ble_time_us=1_000_000 + i * 5_000,
                spacing_us=5_000,
                samples=[[i, 0, 0, 0, 0, 0]],
            )
            device._notification_handler(None, packet)

        self.assertEqual(len(all_dfs), 3)
        self.assertEqual(len(device.df), 3)


class TestGetLatest(unittest.TestCase):

    def test_returns_none_when_no_data(self):
        device = _build_device()
        device.df = None
        self.assertIsNone(device.get_latest(10))

    def test_returns_none_on_empty_df(self):
        device = _build_device()
        # df is initialized but empty
        self.assertIsNone(device.get_latest(10))

    def test_returns_last_n_rows(self):
        device = _build_device()
        for i in range(10):
            packet = _build_notification(
                ble_time_us=1_000_000 + i * 5_000,
                spacing_us=5_000,
                samples=[[i, 0, 0, 0, 0, 0]],
            )
            device._notification_handler(None, packet)

        result = device.get_latest(3)
        self.assertEqual(len(result), 3)

    def test_returns_all_rows_when_n_exceeds_available(self):
        device = _build_device()
        for i in range(5):
            packet = _build_notification(
                ble_time_us=1_000_000 + i * 5_000,
                spacing_us=5_000,
                samples=[[i, 0, 0, 0, 0, 0]],
            )
            device._notification_handler(None, packet)

        result = device.get_latest(100)
        self.assertEqual(len(result), 5)

    def test_has_t_and_sensor_columns(self):
        device = _build_device()
        packet = _build_notification(
            ble_time_us=1_000_000,
            spacing_us=5_000,
            samples=[[1, 2, 3, 4, 5, 6]],
        )
        device._notification_handler(None, packet)

        result = device.get_latest(1)
        expected_cols = {"T", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"}
        self.assertEqual(set(result.columns), expected_cols)

    def test_applies_calibration(self):
        device = _build_device()
        device.calibration = {"acc_x": {"coef": 3.0, "offset": -10.0}}

        raw_acc_x = 200
        packet = _build_notification(
            ble_time_us=1_000_000,
            spacing_us=5_000,
            samples=[[raw_acc_x, 0, 0, 0, 0, 0]],
        )
        device._notification_handler(None, packet)

        result = device.get_latest(1)
        expected = 3.0 * (raw_acc_x * ACC_FACTOR + (-10.0))
        self.assertAlmostEqual(result["acc_x"].iloc[0], expected, places=6)

    def test_t_increases_over_time(self):
        device = _build_device()
        for i in range(5):
            packet = _build_notification(
                ble_time_us=1_000_000 + i * 100_000,
                spacing_us=5_000,
                samples=[[0, 0, 0, 0, 0, 0]],
            )
            device._notification_handler(None, packet)

        result = device.get_latest(5)
        t_values = result["T"].tolist()
        self.assertTrue(all(t_values[i] <= t_values[i + 1] for i in range(len(t_values) - 1)))

    def test_does_not_mutate_internal_df(self):
        device = _build_device()
        packet = _build_notification(
            ble_time_us=1_000_000,
            spacing_us=5_000,
            samples=[[100, 0, 0, 0, 0, 0]],
        )
        device._notification_handler(None, packet)

        before_cols = list(device.df.columns)
        device.get_latest(1)
        self.assertEqual(list(device.df.columns), before_cols)
        self.assertNotIn("T", device.df.columns)


if __name__ == "__main__":
    unittest.main()
