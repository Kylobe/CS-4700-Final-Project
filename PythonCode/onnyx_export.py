import torch
from AlphaZeroChess import AlphaZeroChess

@torch.no_grad()
def export_onnx(model: torch.nn.Module, onnx_path: str):
    model.eval()

    # ONNX export is usually simplest on CPU to avoid CUDA/export quirks
    model_cpu = model.to("cpu")

    dummy = torch.randn(1, 17, 8, 8, dtype=torch.float32, device="cpu")

    # (Optional but recommended) sanity-check output shapes
    policy_logits, value = model_cpu(dummy)
    print("policy_logits:", tuple(policy_logits.shape))  # expect (1, 73, 8, 8) or (73, 8, 8)
    print("value:", tuple(value.shape))                  # expect (1,) or (1,1)

    torch.onnx.export(
        model_cpu,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["state"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "state": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
    )
    print(f"Exported to {onnx_path}")


def main():
    config = {
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "res_blocks": 40,
        "num_hidden": 256,
        "batch_size": 64,
        "epochs": 50,
    }

    model = AlphaZeroChess(
        num_resBlocks=config["res_blocks"],
        num_hidden=config["num_hidden"]
    )

    # Load weights onto CPU (cleanest for export)
    state = torch.load("FiftyMillionPos.pt", map_location="cpu")
    model.load_state_dict(state)

    export_onnx(model, "fifty_million_chess_model.onnx")


if __name__ == "__main__":
    main()