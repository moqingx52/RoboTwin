from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _read_instruction(path: Path) -> str:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    seen = payload.get("seen", [])
    if seen:
        return str(seen[0])
    unseen = payload.get("unseen", [])
    return str(unseen[0]) if unseen else ""


def _episode_id(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 0


def build_embeddings(
    instruction_dir: Path,
    output_npz: Path,
    encoder_path: str,
    tokenizer_max_length: int = 128,
    device: str = "cuda",
) -> None:
    from policy.RDT.multimodal_encoder.t5_encoder import T5Embedder

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    embedder = T5Embedder(
        from_pretrained=encoder_path,
        model_max_length=tokenizer_max_length,
        device=dev,
        use_offload_folder=None,
    )
    tokenizer = embedder.tokenizer
    text_encoder = embedder.model
    text_encoder.eval()

    rows: dict[str, np.ndarray] = {}
    for path in sorted(instruction_dir.glob("episode*.json")):
        episode = _episode_id(path)
        text = _read_instruction(path).strip()
        if not text:
            continue
        with torch.no_grad():
            token_ids = tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
            )["input_ids"].to(dev)
            emb = text_encoder(token_ids).last_hidden_state.detach().to(torch.float32)
        rows[f"episode{episode}"] = emb.cpu().numpy()

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="为 RoboTwin 数据离线预计算语言 embedding。")
    parser.add_argument("--instruction-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--encoder-path",
        type=str,
        default="weights/RDT/t5-v1_1-xxl",
    )
    parser.add_argument("--tokenizer-max-length", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    build_embeddings(
        instruction_dir=Path(args.instruction_dir),
        output_npz=Path(args.output),
        encoder_path=args.encoder_path,
        tokenizer_max_length=int(args.tokenizer_max_length),
        device=args.device,
    )


if __name__ == "__main__":
    main()
