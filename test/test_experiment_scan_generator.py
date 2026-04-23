import unittest

import numpy as np

from ndscan.experiment.scan_generator import (
    CentreSpanGenerator,
    CentreSpanRefiningGenerator,
    ExpandingGenerator,
    LinearGenerator,
    ListGenerator,
    RefiningGenerator,
    ScanOptions,
    generate_points,
)


class ScanGeneratorCase(unittest.TestCase):
    def test_expanding_trivial(self):
        gen_args = {
            "centre": 0.0,
            "spacing": 10.0,
            "randomise_order": False,
            "limit_lower": -1.0,
            "limit_upper": 1.0,
        }

        gen_float = ExpandingGenerator(**gen_args)
        gen_int = ExpandingGenerator(**gen_args, dtype="int")

        for gen in (gen_float, gen_int):
            self.assertTrue(gen.has_level(0))
            self.assertEqual(gen.points_for_level(0), [0.0])
            for i in range(1, 3):
                self.assertFalse(gen.has_level(i))

    def test_expanding_one(self):
        gen_args = {
            "centre": 0.0,
            "spacing": 1.0,
            "randomise_order": False,
            "limit_lower": -1.0,
            "limit_upper": 1.0,
        }

        gen_float = ExpandingGenerator(**gen_args)
        gen_int = ExpandingGenerator(**gen_args, dtype="int")

        for gen in (gen_float, gen_int):
            self.assertTrue(gen.has_level(0))
            self.assertEqual(gen.points_for_level(0), [0.0])
            self.assertTrue(gen.has_level(1))
            self.assertEqual(gen.points_for_level(1), [-1.0, 1.0])
            self.assertFalse(gen.has_level(2))

    def test_expanding_lower_lim(self):
        gen_args = {
            "centre": 0.0,
            "spacing": 1.0,
            "randomise_order": False,
            "limit_lower": -1.0,
        }
        gen_float = ExpandingGenerator(**gen_args)
        gen_int = ExpandingGenerator(**gen_args, dtype="int")

        for gen in (gen_float, gen_int):
            self.assertTrue(gen.has_level(0))
            self.assertEqual(gen.points_for_level(0), [0.0])
            self.assertTrue(gen.has_level(1))
            self.assertEqual(gen.points_for_level(1), [-1.0, 1.0])
            self.assertTrue(gen.has_level(2))
            self.assertEqual(gen.points_for_level(2), [2.0])
            self.assertTrue(gen.has_level(10))
            self.assertEqual(gen.points_for_level(10), [10.0])

    def test_expanding_upper_lim(self):
        gen_args = {
            "centre": 0.0,
            "spacing": 1.0,
            "randomise_order": False,
            "limit_upper": 1.0,
        }
        gen_float = ExpandingGenerator(**gen_args)
        gen_int = ExpandingGenerator(**gen_args, dtype="int")

        for gen in (gen_float, gen_int):
            self.assertTrue(gen.has_level(0))
            self.assertEqual(gen.points_for_level(0), [0.0])
            self.assertTrue(gen.has_level(1))
            self.assertEqual(gen.points_for_level(1), [-1.0, 1.0])
            self.assertTrue(gen.has_level(2))
            self.assertEqual(gen.points_for_level(2), [-2.0])
            self.assertTrue(gen.has_level(10))
            self.assertEqual(gen.points_for_level(10), [-10.0])

    def test_expanding_no_lim(self):
        gen_args = {
            "centre": 0.0,
            "spacing": 1.0,
            "randomise_order": False,
        }
        gen_float = ExpandingGenerator(**gen_args)
        gen_int = ExpandingGenerator(**gen_args, dtype="int")

        for gen in (gen_float, gen_int):
            self.assertTrue(gen.has_level(0))
            self.assertEqual(gen.points_for_level(0), [0.0])
            self.assertTrue(gen.has_level(1))
            self.assertEqual(gen.points_for_level(1), [-1.0, 1.0])
            self.assertTrue(gen.has_level(10))
            self.assertEqual(gen.points_for_level(10), [-10.0, 10.0])

    def test_expanding_int_with_nonint_spacing(self):
        spacing = 1.7

        with self.assertRaises(ValueError):
            ExpandingGenerator(
                centre=0.0, spacing=spacing, randomise_order=False, dtype="int"
            )

    def test_centre_empty(self):
        with self.assertRaises(ValueError):
            CentreSpanGenerator(
                centre=0.0, half_span=1.0, num_points=0, randomise_order=False
            )

    def test_centre_one(self):
        gen_args = {
            "centre": 0.0,
            "half_span": 1.0,
            "num_points": 1,
            "randomise_order": True,
        }
        gen_float = CentreSpanGenerator(**gen_args)
        gen_int = CentreSpanGenerator(**gen_args, dtype="int")

        for gen in (gen_float, gen_int):
            self.assertTrue(gen.has_level(0))
            self.assertEqual(gen.points_for_level(0, np.random), [0.0])
            self.assertFalse(gen.has_level(1))

    def test_centre_two(self):
        gen_args = {
            "centre": 0.0,
            "half_span": 1.0,
            "num_points": 2,
            "randomise_order": False,
        }
        gen_float = CentreSpanGenerator(**gen_args)
        gen_int = CentreSpanGenerator(**gen_args, dtype="int")

        for gen in (gen_float, gen_int):
            self.assertTrue(gen.has_level(0))
            self.assertEqual(gen.points_for_level(0), [-1.0, 1.0])
            self.assertFalse(gen.has_level(1))

    def test_centre_three(self):
        gen_args = {
            "centre": 0.0,
            "half_span": 1.0,
            "num_points": 3,
            "randomise_order": False,
        }
        gen_float = CentreSpanGenerator(**gen_args)
        gen_int = CentreSpanGenerator(**gen_args, dtype="int")

        for gen in (gen_float, gen_int):
            self.assertTrue(gen.has_level(0))
            self.assertEqual(gen.points_for_level(0), [-1.0, 0.0, 1.0])
            self.assertFalse(gen.has_level(1))

    def test_centre_lower_lim(self):
        gen_args = {
            "centre": 0.0,
            "half_span": 1.0,
            "num_points": 2,
            "randomise_order": False,
            "limit_lower": 0.0,
        }
        gen_float = CentreSpanGenerator(**gen_args)
        gen_int = CentreSpanGenerator(**gen_args, dtype="int")
        for gen in (gen_float, gen_int):
            self.assertTrue(gen.has_level(0))
            self.assertEqual(gen.points_for_level(0), [0.0, 1.0])
            self.assertFalse(gen.has_level(1))

    def test_centre_upper_lim(self):
        gen_args = {
            "centre": 0.0,
            "half_span": 1.0,
            "num_points": 2,
            "randomise_order": False,
            "limit_upper": 0.0,
        }
        gen_float = CentreSpanGenerator(**gen_args)
        gen_int = CentreSpanGenerator(**gen_args, dtype="int")
        for gen in (gen_float, gen_int):
            self.assertTrue(gen.has_level(0))
            self.assertEqual(gen.points_for_level(0), [-1.0, 0.0])
            self.assertFalse(gen.has_level(1))

    def test_centre_int_with_nonint_half_span(self):
        with self.assertRaises(ValueError):
            half_span = 1.7
            CentreSpanGenerator(
                centre=0.0,
                half_span=half_span,
                num_points=2,
                randomise_order=False,
                dtype="int",
            )

    def test_centre_int_with_too_many_points(self):
        base_args = {
            "centre": 0.0,
            "half_span": 1.0,
            "num_points": 5,
            "randomise_order": False,
            "dtype": "int",
        }

        with self.assertLogs(level="WARNING"):
            gen = CentreSpanGenerator(**base_args)
            self.assertEqual(gen.num_points, 3.0)
            self.assertEqual(gen.points_for_level(0), [-1, 0, 1])

        with self.assertLogs(level="WARNING"):
            gen = CentreSpanGenerator(**base_args, limit_lower=0.0)
            self.assertEqual(gen.num_points, 2.0)
            self.assertEqual(gen.points_for_level(0), [0, 1])

        with self.assertLogs(level="WARNING"):
            gen = CentreSpanGenerator(**base_args, limit_upper=0.0)
            self.assertEqual(gen.num_points, 2.0)
            self.assertEqual(gen.points_for_level(0), [-1, 0])

    def test_linear_basic(self):
        gen_args = {
            "start": 0.0,
            "stop": 2.0,
            "num_points": 3,
            "randomise_order": False,
        }
        gen_float = LinearGenerator(**gen_args)
        gen_int = LinearGenerator(**gen_args, dtype="int")

        for gen in (gen_float, gen_int):
            self.assertTrue(gen.has_level(0))
            self.assertEqual(gen.points_for_level(0), [0.0, 1.0, 2.0])
            self.assertFalse(gen.has_level(1))

    def test_linear_one_point(self):
        with self.assertRaises(ValueError):
            LinearGenerator(start=5.0, stop=5.0, num_points=1, randomise_order=False)

        with self.assertRaises(ValueError):
            LinearGenerator(
                start=5.0, stop=5.0, num_points=1, randomise_order=False, dtype="int"
            )

    def test_linear_int_too_many_points(self):
        with self.assertLogs(level="WARNING"):
            gen = LinearGenerator(
                start=0.0, stop=2.0, num_points=5, randomise_order=False, dtype="int"
            )
            self.assertEqual(gen.num_points, 3.0)
            self.assertEqual(gen.points_for_level(0), [0, 1, 2])

    def test_listgenerator(self):
        gen = ListGenerator(values=[1.0, 2.0, 3.0], randomise_order=False)
        self.assertTrue(gen.has_level(0))
        self.assertEqual(gen.points_for_level(0), [1.0, 2.0, 3.0])
        self.assertFalse(gen.has_level(1))

    def test_refininggenerator(self):
        gen = RefiningGenerator(lower=0.0, upper=2.0, randomise_order=False)
        self.assertTrue(gen.has_level(0))
        self.assertEqual(gen.points_for_level(0), [0.0, 2.0])
        self.assertTrue(gen.has_level(1))
        self.assertEqual(gen.points_for_level(1), [1.0])
        self.assertTrue(gen.has_level(2))
        self.assertEqual(gen.points_for_level(2), [0.5, 1.5])
        self.assertTrue(gen.has_level(3))
        self.assertEqual(gen.points_for_level(3), [0.25, 0.75, 1.25, 1.75])

    def test_refininggenerator_int(self):
        gen = RefiningGenerator(
            lower=0.0, upper=2.0, randomise_order=False, dtype="int"
        )
        self.assertTrue(gen.has_level(0))
        self.assertEqual(gen.points_for_level(0), [0, 2])
        self.assertTrue(gen.has_level(1))
        self.assertEqual(gen.points_for_level(1), [1])
        self.assertFalse(gen.has_level(2))

        gen = RefiningGenerator(
            lower=1.0, upper=4.0, randomise_order=False, dtype="int"
        )
        self.assertTrue(gen.has_level(0))
        self.assertEqual(gen.points_for_level(0), [1, 4])
        self.assertTrue(gen.has_level(1))
        self.assertTrue(gen.has_level(2))
        self.assertFalse(gen.has_level(3))
        # We don't care whether 2 or 3 comes first or which ones are
        # duplicated in level 2
        points_levels_1_2 = set(gen.points_for_level(1) + gen.points_for_level(2))
        self.assertIn(2, points_levels_1_2)
        self.assertIn(3, points_levels_1_2)
        all_points = points_levels_1_2.union(gen.points_for_level(0))
        self.assertEqual(all_points, {1, 2, 3, 4})

        gen = RefiningGenerator(
            lower=1.0, upper=7.0, randomise_order=False, dtype="int"
        )
        self.assertTrue(gen.has_level(0))
        self.assertEqual(gen.points_for_level(0), [1, 7])
        self.assertTrue(gen.has_level(1))
        self.assertEqual(gen.points_for_level(1), [4])
        self.assertTrue(gen.has_level(2))
        self.assertTrue(gen.has_level(3))
        self.assertFalse(gen.has_level(4))
        points_levels_2_3 = set(gen.points_for_level(2) + gen.points_for_level(3))
        for point in (2, 3, 5, 6):
            self.assertIn(point, points_levels_2_3)
        all_points = points_levels_2_3.union(gen.points_for_level(0)).union(
            gen.points_for_level(1)
        )
        self.assertEqual(all_points, set(range(1, 8)))

    def test_centrespanrefininggenerator(self):
        gen_args = {
            "centre": 200.0,
            "half_span": 100.0,
            "randomise_order": False,
        }
        gen_float = CentreSpanRefiningGenerator(**gen_args)
        gen_int = CentreSpanRefiningGenerator(**gen_args, dtype="int")
        for gen in (gen_float, gen_int):
            self.assertTrue(gen.has_level(0))
            self.assertEqual(gen.points_for_level(0), [100.0, 300.0])
            self.assertTrue(gen.has_level(1))
            self.assertEqual(gen.points_for_level(1), [200.0])
            self.assertTrue(gen.has_level(2))
            self.assertEqual(gen.points_for_level(2), [150.0, 250.0])
            self.assertTrue(gen.has_level(3))
            self.assertEqual(gen.points_for_level(3), [125.0, 175.0, 225.0, 275.0])

    def test_centrespanrefininggenerator_lower_lim(self):
        gen_args = {
            "centre": 200.0,
            "half_span": 100.0,
            "randomise_order": False,
            "limit_lower": 120.0,
        }

        gen_float = CentreSpanRefiningGenerator(**gen_args)
        gen_int = CentreSpanRefiningGenerator(
            **gen_args,
            dtype="int",
        )

        for gen in (gen_float, gen_int):
            self.assertTrue(gen.has_level(0))
            self.assertEqual(gen.points_for_level(0), [300.0])
            self.assertTrue(gen.has_level(1))
            self.assertEqual(gen.points_for_level(1), [200.0])
            self.assertTrue(gen.has_level(2))
            self.assertEqual(gen.points_for_level(2), [150.0, 250.0])
            self.assertTrue(gen.has_level(3))
            self.assertEqual(gen.points_for_level(3), [125.0, 175.0, 225.0, 275.0])

    def test_centrespanrefininggenerator_span_outside_limit(self):
        gen_args = {
            "centre": 200.0,
            "half_span": 100.0,
            "randomise_order": False,
            "limit_lower": 120.0,
            "limit_upper": 250.0,
        }
        gen_float = CentreSpanRefiningGenerator(**gen_args)
        gen_int = CentreSpanRefiningGenerator(**gen_args, dtype="int")

        for gen in (gen_float, gen_int):
            self.assertTrue(gen.has_level(0))
            self.assertEqual(gen.points_for_level(0), [120.0])
            self.assertTrue(gen.has_level(1))
            self.assertEqual(gen.points_for_level(1), [200.0])
            self.assertTrue(gen.has_level(2))
            self.assertEqual(gen.points_for_level(2), [160.0, 240.0])
            self.assertTrue(gen.has_level(3))
            self.assertEqual(gen.points_for_level(3), [140.0, 180.0, 220.0])

    def test_centrespanrefininggenerator_int(self):
        gen = CentreSpanRefiningGenerator(
            centre=10.0,
            half_span=3.0,
            randomise_order=False,
            limit_lower=9.0,
            limit_upper=13.0,
            dtype="int",
        )

        self.assertTrue(gen.has_level(0))
        self.assertEqual(gen.points_for_level(0), [13])
        self.assertTrue(gen.has_level(1))
        self.assertEqual(gen.points_for_level(1), [10])
        self.assertTrue(gen.has_level(2))
        self.assertTrue(gen.has_level(3))

        points_levels_2_3 = set(gen.points_for_level(2) + gen.points_for_level(3))
        for point in (9, 11, 12):
            self.assertIn(point, points_levels_2_3)
        all_points = points_levels_2_3.union(gen.points_for_level(0)).union(
            gen.points_for_level(1)
        )
        self.assertEqual(all_points, set(range(9, 14)))


class GeneratePointsCase(unittest.TestCase):
    def test_no_repeats(self):
        opt = ScanOptions()
        gen = CentreSpanGenerator(
            centre=0.0, half_span=1.0, num_points=2, randomise_order=False
        )
        points = list(generate_points([gen], opt))
        self.assertEqual(points, [(-1.0,), (1.0,)])

    def test_repeat_scan(self):
        opt = ScanOptions(num_repeats=2)
        gen = CentreSpanGenerator(
            centre=0.0, half_span=1.0, num_points=2, randomise_order=False
        )
        points = list(generate_points([gen], opt))
        self.assertEqual(points, [(-1.0,), (1.0,), (-1.0,), (1.0,)])

    def test_repeat_each_point(self):
        opt = ScanOptions(num_repeats=1, num_repeats_per_point=2)
        gen = CentreSpanGenerator(
            centre=0.0, half_span=1.0, num_points=2, randomise_order=False
        )
        points = list(generate_points([gen], opt))
        self.assertEqual(points, [(-1.0,), (-1.0,), (1.0,), (1.0,)])

    def test_repeat_scan_and_each_point(self):
        opt = ScanOptions(num_repeats=2, num_repeats_per_point=2)
        gen = CentreSpanGenerator(
            centre=0.0, half_span=1.0, num_points=2, randomise_order=False
        )
        points = list(generate_points([gen], opt))
        self.assertEqual(
            points, [(-1.0,), (-1.0,), (1.0,), (1.0,), (-1.0,), (-1.0,), (1.0,), (1.0,)]
        )

    def test_2d_scan(self):
        opt = ScanOptions(num_repeats=1, num_repeats_per_point=1)
        gen1 = CentreSpanGenerator(
            centre=0.0, half_span=1.0, num_points=2, randomise_order=False
        )
        gen2 = CentreSpanGenerator(
            centre=0.0, half_span=20.0, num_points=2, randomise_order=False
        )
        points = list(generate_points([gen1, gen2], opt))
        self.assertEqual(
            points, [(-1.0, -20.0), (1.0, -20.0), (-1.0, 20.0), (1.0, 20.0)]
        )
