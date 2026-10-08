"""Synthetic arithmetic examples only; no model downloads or API calls."""
def weight_gb(params_b, bits):
    return params_b * bits / 8


def kv_gib(layers, requests, tokens, kv_heads, head_dim, dtype_bytes):
    return (2 * layers * requests * tokens * kv_heads
            * head_dim * dtype_bytes) / 2**30


def api_cost(input_tokens, output_tokens, input_price, output_price):
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


print(f"8B / 4bit 가중치: {weight_gb(8, 4):.1f} GB")
print(f"혼합 정밀도 가중치: {weight_gb(7, 4) + weight_gb(1, 16):.1f} GB")
print(f"8개 요청 KV 캐시: {kv_gib(32, 8, 4096, 8, 128, 2):.1f} GiB")
print(f"가상 단가의 100건 비용: ${api_cost(600_000, 50_000, 0.8, 2.4):.2f}")
