import pytest
import torch
from sgl_kernel import blockmask_to_uint64, topk_to_uint64, uint64_to_bool


@pytest.mark.parametrize("shape", [(2, 5, 130), (3, 64), (1, 4, 200), (2, 63)])
def test_blockmask_uint64_roundtrip(shape):
    """blockmask -> uint64 -> bool must reproduce the original mask exactly."""
    torch.manual_seed(0)
    last_dim = shape[-1]
    mask = torch.rand(shape, device="cuda") > 0.5  # bool

    packed, ld = blockmask_to_uint64(mask)
    assert ld == last_dim
    assert packed.shape[-1] == (last_dim + 63) // 64
    assert packed.shape[:-1] == mask.shape[:-1]

    restored = uint64_to_bool(packed, last_dim)
    assert torch.equal(restored, mask)


@pytest.mark.parametrize("num_heads,total_seqlen,k", [(2, 7, 4), (1, 3, 8)])
def test_topk_to_uint64_matches_bitset(num_heads, total_seqlen, k):
    """topk indices packed to uint64 then expanded to bool must match a scatter."""
    torch.manual_seed(0)
    max_seqlen_k = 512
    block_size = 64
    k_blocks = (max_seqlen_k + block_size - 1) // block_size  # 8

    # Random distinct block ids in [0, k_blocks); pad with -1.
    topk = torch.full((num_heads, total_seqlen, k), -1, dtype=torch.int32)
    for h in range(num_heads):
        for s in range(total_seqlen):
            n = torch.randint(0, k + 1, (1,)).item()
            if n > 0:
                ids = torch.randperm(k_blocks)[:n]
                topk[h, s, : ids.numel()] = ids.to(torch.int32)
    topk = topk.cuda()

    packed, kb = topk_to_uint64(topk, max_seqlen_k, block_size)
    assert kb == k_blocks

    restored = uint64_to_bool(packed, k_blocks)

    # Reference bool mask from scatter.
    ref = torch.zeros(
        num_heads, total_seqlen, k_blocks, dtype=torch.bool, device="cuda"
    )
    flat_topk = topk.reshape(-1, k)
    flat_ref = ref.reshape(-1, k_blocks)
    for row in range(flat_topk.shape[0]):
        for j in range(k):
            idx = int(flat_topk[row, j].item())
            if idx != -1:
                flat_ref[row, idx] = True
    assert torch.equal(restored, ref)


def test_uint64_to_bool_known_pattern():
    """A single uint64 with bits 0 and 63 set expands correctly."""
    val = (1 << 0) | (1 << 63)
    # Store as signed int64 bit pattern.
    signed = val - (1 << 64) if val >= (1 << 63) else val
    packed = torch.tensor([[signed]], dtype=torch.int64, device="cuda")
    out = uint64_to_bool(packed, 64)
    expected = torch.zeros(1, 64, dtype=torch.bool, device="cuda")
    expected[0, 0] = True
    expected[0, 63] = True
    assert torch.equal(out, expected)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
