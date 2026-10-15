import unittest
from memory_selection_v1 import MemoryRecord, PoseMemoryIndex, SelectionPolicy, angular_distance_degrees


def record(start, *, name=None, world="w", map_id="m", xyz=(0, 0, 0), yaw=0):
    return MemoryRecord(name or str(start), world, map_id, start, 3, xyz, yaw,
                        "latent.pt", "a" * 64, "camera.pt", "b" * 64)


def query(index, **kwargs):
    args = dict(world_id="w", map_revision_id="m", current_latent_start=60,
                position_xyz=(0, 0, 0), yaw_degrees=0)
    args.update(kwargs)
    return index.query(**args)


class MemorySelectionTests(unittest.TestCase):
    def test_old_same_view_selected(self):
        bank = PoseMemoryIndex(); bank.commit(record(3))
        selected, audit = query(bank)
        self.assertEqual(selected.record_id, "3")
        self.assertEqual(audit["distance_m"], 0)

    def test_wrong_map_world_recent_opposite_and_distant_rejected(self):
        cases = [record(3, world="other"), record(3, map_id="other"), record(48),
                 record(3, yaw=180), record(3, xyz=(3, 0, 0))]
        for candidate in cases:
            bank = PoseMemoryIndex(); bank.commit(candidate)
            self.assertIsNone(query(bank)[0])

    def test_earliest_tie_is_deterministic_and_no_quality_selection(self):
        bank = PoseMemoryIndex(); bank.commit(record(3)); bank.commit(record(6))
        self.assertEqual(query(bank)[0].record_id, "3")

    def test_angle_wrap(self):
        self.assertEqual(angular_distance_degrees(179, -179), 2)

    def test_bound_and_duplicate_nonoverlap(self):
        bank = PoseMemoryIndex(SelectionPolicy(max_records=2))
        bank.commit(record(3)); bank.commit(record(6))
        self.assertEqual(bank.commit(record(9)), ["3"])
        with self.assertRaises(ValueError): bank.commit(record(9))
        with self.assertRaises(ValueError): bank.commit(record(10))

    def test_i0_or_non_native_chunk_rejected(self):
        with self.assertRaises(ValueError): record(0)


if __name__ == "__main__":
    unittest.main()
