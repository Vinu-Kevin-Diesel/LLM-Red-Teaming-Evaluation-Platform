"""Export a student checkpoint for CPU serving: ONNX fp32, dynamic INT8, tokenizer, config.

    python -m llm_eval_forge.serve.export models/student-AB --out artifacts/student-AB

The artifact directory holds everything the ONNX runtime needs, so the serving image needs
neither torch nor transformers:
    model.onnx        fp32 graph (opset 17, dynamic batch and sequence axes)
    model.int8.onnx   dynamic INT8, with the MLP output projections kept in fp32
    tokenizer.json    fast tokenizer (pair encoding, truncation applied at runtime)
    serve_config.json thresholds and metadata
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

METRICS_DIR = Path("results/metrics")


def serve_config(model_dir: Path, name: str, max_len: int = 2048) -> dict:
    """Decision thresholds come from the evaluation run: `teacher` matches Claude's verdicts
    (max kappa on the teacher-labeled split), `gold` maximizes F1 on public labels."""
    metrics_path = METRICS_DIR / f"{name}.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return {
        "model_version": name,
        "source_checkpoint": str(model_dir),
        "max_len": max_len,
        "thresholds": {
            "teacher": metrics.get("teacher_threshold", 0.5),
            "gold": metrics.get("threshold", 0.5),
        },
        "temperature": metrics.get("temperature", 1.0),
        "uncertain_band": [0.2, 0.8],
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }


def export_onnx(model_dir: Path, out_dir: Path) -> Path:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from llm_eval_forge.distill.train import format_pair

    # Eager attention traces cleanly; SDPA's fused kernels are not needed for the CPU graph.
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    enc = tok(*format_pair("How do I bake bread?", "Mix flour, water, yeast and salt."), return_tensors="pt")
    path = out_dir / "model.onnx"
    torch.onnx.export(
        model, (enc["input_ids"], enc["attention_mask"]), str(path),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    tok.backend_tokenizer.save(str(out_dir / "tokenizer.json"))
    return path


def mlp_output_projections(fp32_path: Path) -> list[str]:
    import onnx

    graph = onnx.load(str(fp32_path), load_external_data=False).graph
    return [n.name for n in graph.node if n.op_type == "MatMul" and "/mlp/Wo/" in n.name]


def quantize_int8(fp32_path: Path) -> tuple[Path, dict]:
    """Dynamic INT8 for every MatMul except each MLP's output projection (Wo).

    Plain dynamic INT8 flips ~11% of ModernBERT's decisions (base) and ~20% (large): the
    post-GeGLU activations feeding Wo have outliers that per-tensor activation quantization
    clips. Keeping only those 22-28 MatMuls in fp32 brings flips to ~1% and stays ~1.9x faster
    than fp32 on CPU (measured in results/latency.md)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out = fp32_path.with_name("model.int8.onnx")
    keep_fp32 = mlp_output_projections(fp32_path)
    quantize_dynamic(str(fp32_path), str(out), weight_type=QuantType.QInt8, nodes_to_exclude=keep_fp32)
    return out, {"method": "onnxruntime dynamic INT8 (QInt8 weights)", "fp32_nodes": "mlp/Wo MatMul",
                 "fp32_node_count": len(keep_fp32)}


def parity_check(model_dir: Path, out_dir: Path, n: int = 256, threshold: float = 0.5) -> dict:
    """Compare torch fp32 with ONNX fp32 / INT8 on real test rows."""
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from llm_eval_forge.distill.data import read_split
    from llm_eval_forge.distill.train import format_pair
    from llm_eval_forge.serve.runtime import OnnxRuntime

    rows = read_split(Path("data"), "test", {"harmbench_val", "wildguardtest", "beavertails_test"})
    random.Random(0).shuffle(rows)
    rows = rows[:n]
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, attn_implementation="sdpa").eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    ref = []
    with torch.inference_mode():
        for r in rows:
            enc = tok(*format_pair(r["prompt"], r["response"]), truncation="longest_first", max_length=2048,
                      return_tensors="pt")
            ref.append(model(**enc).logits.softmax(-1)[0, 1].item())
    out = {"n": n}
    for name in ("model.onnx", "model.int8.onnx"):
        rt = OnnxRuntime(out_dir, onnx_file=name)
        got = rt.predict([(r["prompt"], r["response"]) for r in rows])
        diff = np.abs(np.array(got) - np.array(ref))
        out[name] = {"max_abs_dp": round(float(diff.max()), 5), "mean_abs_dp": round(float(diff.mean()), 5),
                     "label_agreement_at_threshold": round(float(np.mean(
                         (np.array(got) >= threshold) == (np.array(ref) >= threshold))), 4)}
    out["threshold"] = threshold
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a student checkpoint to ONNX (+INT8)")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default=None, help="Metrics name (default: checkpoint dir name)")
    args = parser.parse_args()

    name = args.name or args.model_dir.name
    args.out.mkdir(parents=True, exist_ok=True)
    fp32 = export_onnx(args.model_dir, args.out)
    int8, recipe = quantize_int8(fp32)
    cfg = serve_config(args.model_dir, name)
    cfg["quantization"] = recipe
    for f in ("config.json",):
        shutil.copy(args.model_dir / f, args.out / f)
    parity = parity_check(args.model_dir, args.out, threshold=cfg["thresholds"]["teacher"])
    cfg["parity_vs_torch_fp32"] = parity
    cfg["sizes_mb"] = {p.name: round(p.stat().st_size / 2**20, 1) for p in (fp32, int8)}
    (args.out / "serve_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(json.dumps({k: cfg[k] for k in ("model_version", "thresholds", "sizes_mb", "parity_vs_torch_fp32")}, indent=2))


if __name__ == "__main__":
    main()
