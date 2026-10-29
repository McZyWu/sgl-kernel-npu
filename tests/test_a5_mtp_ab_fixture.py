"""CPU-only checks for the deterministic A5 MTP fixture helpers."""

import unittest

from benchmark.lim_a5_mtp_ab.fixture import (
    build_residency,
    make_matrix,
    target_union_misses,
    validate_causal_boundary,
)


def synthetic_reference():
    return [
        [
            [0, 1, 2, 3],
            [2, 3, 4, 5],
            [4, 5, 6, 7],
            [6, 7, 8, 20],
        ]
    ]


class FixtureTests(unittest.TestCase):
    def test_matrix_has_all_sixteen_cases_and_expected_union_misses(self):
        matrix = make_matrix()
        self.assertEqual(len(matrix), 16)
        self.assertEqual(
            len(
                {
                    (case["mtp"], case["mode"], case["policy"], case["hit_percent"])
                    for case in matrix
                }
            ),
            16,
        )
        self.assertTrue(all(case["q"] == case["mtp"] + 1 for case in matrix))
        self.assertEqual(
            {case["hit_rate_basis"] for case in matrix}, {"selected_prefix_union"}
        )
        self.assertEqual([target_union_misses(1, hit) for hit in (95, 91)], [102, 184])
        self.assertEqual([target_union_misses(4, hit) for hit in (95, 91)], [410, 737])

    def test_residency_keeps_selected_hits_and_misses_stable_across_fillers(self):
        reference = synthetic_reference()
        low = build_residency(reference, 16, 9, 2, "low", 71)
        random_fill = build_residency(reference, 16, 9, 2, "random", 71)
        self.assertEqual(low["missing"], random_fill["missing"])
        self.assertEqual(low["stats"]["selected_prefix_misses"], 2)
        self.assertEqual(low["stats"]["selected_prefix_union"], 9)
        self.assertEqual(low["stats"]["route_occurrences"], 16)
        self.assertEqual(low["stats"]["route_occurrence_misses"], 4)
        self.assertAlmostEqual(low["stats"]["route_occurrence_hit_rate"], 0.75)

        selected_prefix = set(range(9))
        hit_sources = selected_prefix - set(low["missing"][0])
        for source in hit_sources:
            self.assertEqual(low["maps"][0][source], random_fill["maps"][0][source])
        remaining_slots = set(range(7, 9))
        low_filler_slots = {
            low["maps"][0][source]
            for source in low["maps"][0]
            if source not in hit_sources
        }
        random_filler_slots = {
            random_fill["maps"][0][source]
            for source in random_fill["maps"][0]
            if source not in hit_sources
        }
        self.assertEqual(low_filler_slots, remaining_slots)
        self.assertEqual(random_filler_slots, remaining_slots)
        self.assertNotEqual(low["fixture_sha256"], random_fill["fixture_sha256"])

    def test_miss_sets_are_nested_for_95_and_91_percent_with_same_seed(self):
        union = [[list(range(2048))]]
        miss_95 = build_residency(
            union, 4096, 2048, target_union_misses(1, 95), "low", 19
        )
        miss_91 = build_residency(
            union, 4096, 2048, target_union_misses(1, 91), "low", 19
        )
        self.assertEqual(len(miss_95["missing"][0]), 102)
        self.assertEqual(len(miss_91["missing"][0]), 184)
        self.assertTrue(set(miss_95["missing"][0]).issubset(miss_91["missing"][0]))

    def test_tail_sources_are_hits_and_never_enter_prefix_map_or_misses(self):
        result = build_residency(synthetic_reference(), 16, 9, 2, "low", 4)
        self.assertNotIn(20, result["maps"][0])
        self.assertNotIn(20, result["missing"][0])
        self.assertEqual(result["stats"]["requests"][0]["selected_tail_union"], 1)
        self.assertTrue(result["stats"]["tail_sources_are_hits_and_not_resident"])

    def test_causal_boundary_helper_accepts_valid_routes_and_rejects_future_source(
        self,
    ):
        reference = [
            [
                [0, 8],  # Q4 route 0 sees 0..8 for final length 12.
                [1, 9],
                [2, 10],
                [3, 11],
            ]
        ]
        self.assertTrue(validate_causal_boundary(reference, 12))
        reference[0][0][1] = 9
        with self.assertRaisesRegex(ValueError, "causal boundary"):
            validate_causal_boundary(reference, 12)

    def test_invalid_residency_inputs_are_rejected(self):
        invalid_kwargs = [
            {
                "prefix_tokens": 0,
                "capacity": 1,
                "misses": 0,
                "policy": "low",
                "seed": 1,
            },
            {
                "prefix_tokens": 8,
                "capacity": 9,
                "misses": 0,
                "policy": "low",
                "seed": 1,
            },
            {
                "prefix_tokens": 16,
                "capacity": 9,
                "misses": 10,
                "policy": "low",
                "seed": 1,
            },
            {
                "prefix_tokens": 16,
                "capacity": 9,
                "misses": 2,
                "policy": "other",
                "seed": 1,
            },
            {
                "prefix_tokens": 16,
                "capacity": 6,
                "misses": 2,
                "policy": "low",
                "seed": 1,
            },
        ]
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_residency(synthetic_reference(), **kwargs)

    def test_invalid_union_miss_arguments_are_rejected(self):
        for args in [(0, 91, 2048), (1, 101, 2048), (1, 91, 0)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                target_union_misses(*args)

    def test_invalid_nonunique_routes_and_mismatched_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_residency([[[0, 0]]], 4, 2, 0, "low", 0)
        with self.assertRaisesRegex(ValueError, "one value per request"):
            validate_causal_boundary([[[0]], [[0]]], [1])


if __name__ == "__main__":
    unittest.main()
