""" Tests for attn fxns"""

import torch
import pytest
from attention import scaled_dot_product_attention, MultiHeadAttention
from attention import positional_encoding_get_val
import math

def test_scaled_dot_product_attention_basic():
    q = torch.tensor(
        [[[1.0, 0.0],
          [0.0, 1.0]]]
    )
    k = torch.tensor(
        [[[1.0, 0.0],
          [0.0, 1.0]]]
    )
    v = torch.tensor(
        [[[1.0, 2.0],
          [3.0, 4.0]]]
    )

    d_h = 2

    actual = scaled_dot_product_attention(q, k, v, d_h)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    assert actual.shape == (1, 2, 2)
    torch.testing.assert_close(actual, expected)


def test_scaled_dot_product_attention_nontrivial():
    q = torch.tensor(
        [[[1.0, 2.0],
          [3.0, 4.0]]]
    )
    k = torch.tensor(
        [[[2.0, 1.0],
          [0.5, 3.0]]]
    )
    v = torch.tensor(
        [[[1.0, 5.0],
          [2.0, 6.0]]]
    )

    d_h = 2

    actual = scaled_dot_product_attention(q, k, v, d_h)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    assert actual.shape == (1, 2, 2)
    torch.testing.assert_close(actual, expected)


def test_scaled_dot_product_attention_batch():
    q = torch.tensor(
        [
            [[1.0, 0.0],
             [0.0, 1.0]],

            [[1.0, 1.0],
             [2.0, 0.0]],
        ]
    )

    k = torch.tensor(
        [
            [[1.0, 0.0],
             [0.0, 1.0]],

            [[0.0, 1.0],
             [1.0, 0.0]],
        ]
    )

    v = torch.tensor(
        [
            [[1.0, 2.0],
             [3.0, 4.0]],

            [[5.0, 6.0],
             [7.0, 8.0]],
        ]
    )

    d_h = 2

    actual = scaled_dot_product_attention(q, k, v, d_h)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    assert actual.shape == (2, 2, 2)
    torch.testing.assert_close(actual, expected)


def test_scaled_dot_product_attention_longer_sequence():
    q = torch.tensor(
        [[[1.0, 0.0],
          [0.0, 1.0],
          [1.0, 1.0]]]
    )

    k = torch.tensor(
        [[[1.0, 0.0],
          [0.0, 1.0],
          [1.0, 1.0]]]
    )

    v = torch.tensor(
        [[[1.0, 2.0],
          [3.0, 4.0],
          [5.0, 6.0]]]
    )

    d_h = 2

    actual = scaled_dot_product_attention(q, k, v, d_h)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    assert actual.shape == (1, 3, 2)
    torch.testing.assert_close(actual, expected)


def test_scaled_dot_product_attention_larger_embedding():
    q = torch.tensor(
        [[[1.0, 0.0, 1.0, 0.0],
          [0.0, 1.0, 0.0, 1.0]]]
    )

    k = torch.tensor(
        [[[1.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 1.0]]]
    )

    v = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0],
          [5.0, 6.0, 7.0, 8.0]]]
    )

    d_h = 4

    actual = scaled_dot_product_attention(q, k, v, d_h)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    assert actual.shape == (1, 2, 4)
    torch.testing.assert_close(actual, expected)

@pytest.mark.parametrize(
    "q,k,v",
    [
        # Simple 1-batch, 2-token case
        (
            torch.tensor(
                [[[1.0, 0.0, 1.0, 0.0],
                  [0.0, 1.0, 0.0, 1.0]]]
            ),
            torch.tensor(
                [[[1.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 1.0]]]
            ),
            torch.tensor(
                [[[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0]]]
            ),
        ),

        # Negative values
        (
            torch.tensor(
                [[[1.0, -1.0, 2.0, -2.0],
                  [-1.0, 1.0, -2.0, 2.0]]]
            ),
            torch.tensor(
                [[[2.0, -1.0, 0.5, -0.5],
                  [-2.0, 1.0, -0.5, 0.5]]]
            ),
            torch.tensor(
                [[[3.0, -3.0, 1.0, -1.0],
                  [-4.0, 4.0, -2.0, 2.0]]]
            ),
        ),

        # Three-token sequence
        (
            torch.tensor(
                [[[1.0, 2.0, 3.0, 4.0],
                  [4.0, 3.0, 2.0, 1.0],
                  [1.0, 1.0, 1.0, 1.0]]]
            ),
            torch.tensor(
                [[[2.0, 0.0, 1.0, 3.0],
                  [1.0, 2.0, 0.0, 1.0],
                  [0.0, 1.0, 2.0, 1.0]]]
            ),
            torch.tensor(
                [[[1.0, 5.0, 2.0, 6.0],
                  [3.0, 7.0, 4.0, 8.0],
                  [9.0, 1.0, 5.0, 2.0]]]
            ),
        ),

        # Batch size 2
        (
            torch.tensor(
                [
                    [[1.0, 0.0, 1.0, 0.0],
                     [0.0, 1.0, 0.0, 1.0]],

                    [[1.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 1.0]],
                ]
            ),
            torch.tensor(
                [
                    [[1.0, 0.0, 0.0, 1.0],
                     [0.0, 1.0, 1.0, 0.0]],

                    [[2.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 2.0]],
                ]
            ),
            torch.tensor(
                [
                    [[1.0, 2.0, 3.0, 4.0],
                     [5.0, 6.0, 7.0, 8.0]],

                    [[8.0, 7.0, 6.0, 5.0],
                     [4.0, 3.0, 2.0, 1.0]],
                ]
            ),
        ),

        # Zero query
        (
            torch.tensor(
                [[[0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]]]
            ),
            torch.tensor(
                [[[1.0, 2.0, 3.0, 4.0],
                  [4.0, 3.0, 2.0, 1.0]]]
            ),
            torch.tensor(
                [[[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0]]]
            ),
        ),
    ],
)
def test_multi_head_attention_against_torch(q, k, v):
    torch.manual_seed(42)

    d_model = q.shape[-1]
    num_heads = 2

    actual_mha = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
    )

    expected_mha = torch.nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        dropout=0.0,
        bias=False,
        batch_first=True,
    )

    w_q = torch.cat(list(actual_mha.head_q), dim=1).T
    w_k = torch.cat(list(actual_mha.head_k), dim=1).T
    w_v = torch.cat(list(actual_mha.head_v), dim=1).T

    with torch.no_grad():
        expected_mha.in_proj_weight.copy_(
            torch.cat([w_q, w_k, w_v], dim=0)
        )

        expected_mha.out_proj.weight.copy_(
            actual_mha.concat_mat.T
        )

    actual = actual_mha(q, k, v)

    expected, _ = expected_mha(
        q,
        k,
        v,
        need_weights=False,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )

@pytest.mark.parametrize(
    "batch_size,seq_len,d_model,num_heads",
    [
        (1, 2, 4, 1),
        (1, 4, 4, 2),
        (2, 5, 8, 2),
        (3, 7, 8, 4),
        (4, 10, 16, 8),
    ],
)
def test_multi_head_attention_random_against_torch(
    batch_size,
    seq_len,
    d_model,
    num_heads,
):
    torch.manual_seed(42)

    actual_mha = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
    )

    expected_mha = torch.nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        dropout=0.0,
        bias=False,
        batch_first=True,
    )

    w_q = torch.cat(list(actual_mha.head_q), dim=1).T
    w_k = torch.cat(list(actual_mha.head_k), dim=1).T
    w_v = torch.cat(list(actual_mha.head_v), dim=1).T

    with torch.no_grad():
        expected_mha.in_proj_weight.copy_(
            torch.cat([w_q, w_k, w_v], dim=0)
        )
        expected_mha.out_proj.weight.copy_(
            actual_mha.concat_mat.T
        )

    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    actual = actual_mha(q, k, v)

    expected, _ = expected_mha(
        q,
        k,
        v,
        need_weights=False,
    )

    assert actual.shape == (
        batch_size,
        seq_len,
        d_model,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "pos,i,d_model,expected",
    [
        # i = 0 -> sin(pos / 10000^0) = sin(pos)
        (1.0, 0, 8, math.sin(1.0)),

        # i = 1 -> cos(pos / 10000^0) = cos(pos)
        (1.0, 1, 8, math.cos(1.0)),

        # i = 2 -> sin(pos / 10000^(2/8))
        (1.0, 2, 8, math.sin(1.0 / (10000 ** (2 / 8)))),

        # i = 3 uses the same frequency as i = 2
        (1.0, 3, 8, math.cos(1.0 / (10000 ** (2 / 8)))),

        # Different position
        (5.0, 4, 8, math.sin(5.0 / (10000 ** (4 / 8)))),
    ],
)
def test_positional_encoding_get_val(pos, i, d_model, expected):
    result = positional_encoding_get_val(
        torch.tensor(pos),
        i,
        d_model,
    )

    assert torch.isclose(
        result,
        torch.tensor(expected),
        atol=1e-6,
    )
