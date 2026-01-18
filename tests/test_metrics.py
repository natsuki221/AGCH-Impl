import pytest
import torch
import numpy as np
from src.utils.metrics import calculate_hamming_dist_matrix, calculate_mAP, calculate_precision_at_k


# Mark as P0 acceptance tests
@pytest.mark.p0
class TestRetrievalMetrics:
    """Acceptance Tests for Retrieval Metrics (mAP & P@k)"""

    def test_hamming_distance_matrix_correctness(self):
        """
        Acceptance Criterion 3: Compute Hamming distance using vectorized matrix operations.

        Given:
            Query code:  [1, 1, -1]  (length 3)
            Retrieval codes:
                DB1: [1, 1, -1] (Diff=0) Matches exactly
                DB2: [1, -1, -1] (Diff=1) Second bit differs
                DB3: [-1, -1, 1] (Diff=3) All differ

        When:
            calculate_hamming_dist_matrix is called

        Then:
            The distance matrix should be [[0, 1, 3]]
        """
        # Given
        query_code = torch.tensor([[1.0, 1.0, -1.0]])
        retrieval_code = torch.tensor(
            [[1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, -1.0, 1.0]]  # dist=0  # dist=1  # dist=3
        )

        # When
        dist_matrix = calculate_hamming_dist_matrix(query_code, retrieval_code)

        # Then
        assert dist_matrix.shape == (1, 3)
        expected_dist = torch.tensor([[0.0, 1.0, 3.0]])
        assert torch.allclose(
            dist_matrix, expected_dist
        ), f"Expected {expected_dist}, got {dist_matrix}"

    def test_map_calculation_manual(self):
        """
        Acceptance Criterion 5: Return correct mAP value compared to manual calculation.

        Scenario:
            1 Query
            5 Database items sorted by distance: [Correct, Wrong, Correct, Wrong, Wrong]

            Precision at k:
            k=1: 1/1 (Correct) -> P=1.0
            k=2: 1/2 (Wrong)   -> P=0.5
            k=3: 2/3 (Correct) -> P=0.66
            k=4: 2/4 (Wrong)   -> P=0.5
            k=5: 2/5 (Wrong)   -> P=0.4

            AP = (1.0 + 0.66) / 2 = 0.8333...
        """
        # Given: Pre-calculated distance matrix and labels
        # Distances designed to give retrieval order: IDX 0, 1, 2, 3, 4
        dist_matrix = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])

        # Labels: Query=1, DB=[1, 0, 1, 0, 0]
        # Matches indices 0 and 2
        query_labels = torch.tensor(
            [[1, 0, 0]]
        )  # One-hot or just indices, assume function takes labels and compares equality
        # AC implies labels. Let's assume multi-label or single-label.
        # For simplicity and standard hashing, let's assume labels are vectors and match if dot > 0 or share at least one tag.
        # Let's start with simple shared label logic.

        # Let's mock the "match" logic inside the test by providing labels that will "match"
        # Assume input is (N,L) one-hot or multi-hot
        query_labels = torch.tensor([[1.0]])
        retrieval_labels = torch.tensor(
            [
                [1.0],  # Match (Pos 1)
                [0.0],  # No Match (Pos 2)
                [1.0],  # Match (Pos 3)
                [0.0],  # No Match (Pos 4)
                [0.0],  # No Match (Pos 5)
            ]
        )

        # When
        mAP = calculate_mAP(dist_matrix, query_labels, retrieval_labels)

        # Then
        expected_ap = (1.0 + (2.0 / 3.0)) / 2.0
        assert abs(mAP - expected_ap) < 1e-4, f"Expected {expected_ap}, got {mAP}"

    def test_precision_at_k_correctness(self):
        """
        Acceptance Criterion 6: Precision@k matches expected values.

        Using same scenario as above.
        P@1 = 1.0
        P@3 = 0.666
        """
        # Given
        dist_matrix = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
        query_labels = torch.tensor([[1.0]])
        retrieval_labels = torch.tensor([[1.0], [0.0], [1.0], [0.0], [0.0]])

        # When
        p_at_1 = calculate_precision_at_k(dist_matrix, query_labels, retrieval_labels, k=1)
        p_at_3 = calculate_precision_at_k(dist_matrix, query_labels, retrieval_labels, k=3)

        # Then
        assert abs(p_at_1 - 1.0) < 1e-4
        assert abs(p_at_3 - (2.0 / 3.0)) < 1e-4

    def test_vectorized_high_performance_requirement(self):
        """
        Acceptance Criterion 3 (Performance): Ensure solution supports batching.
        Simulate 2 queries, 3 DB items.
        """
        # Given
        # Q1: [1, 1] matches DB1[1, 1], DB2[-1, -1] (inverse?) no, hamming dist
        # Q2: [-1, -1]

        # Q: (2, 2)
        query_code = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])

        # DB: (3, 2)
        retrieval_code = torch.tensor(
            [
                [1.0, 1.0],  # Match Q1 (dist=0), Opp Q2 (dist=2)
                [-1.0, -1.0],  # Opp Q1 (dist=2), Match Q2 (dist=0)
                [1.0, -1.0],  # Mix (dist=1 to both)
            ]
        )

        # When
        dist_matrix = calculate_hamming_dist_matrix(query_code, retrieval_code)

        # Then
        # Q1 distances: [0, 2, 1]
        # Q2 distances: [2, 0, 1]
        expected = torch.tensor([[0.0, 2.0, 1.0], [2.0, 0.0, 1.0]])

        assert dist_matrix.shape == (2, 3)
        assert torch.allclose(dist_matrix, expected)
