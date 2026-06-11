"""
ONNX Export Module for Model Deployment.

Exports PyTorch models to ONNX format with:
- Tracing and scripting export modes
- Dynamic shape support
- ONNX model simplification
- Shape inference
- Opset version management
- Validation and testing
"""

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class ExportConfig:
    """Configuration for ONNX export."""
    opset_version: int = 17
    input_names: List[str] = None
    output_names: List[str] = None
    dynamic_axes: Optional[Dict] = None
    do_constant_folding: bool = True
    simplify: bool = True
    check_model: bool = True
    test_export: bool = True
    external_data: bool = False  # For large models >2GB
    export_mode: str = "trace"  # trace, script

    def __post_init__(self):
        if self.input_names is None:
            self.input_names = ["input"]
        if self.output_names is None:
            self.output_names = ["output"]
        if self.dynamic_axes is None:
            self.dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }


def export_to_onnx(
    model: Any,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    config: Optional[ExportConfig] = None,
    dynamic_shapes: bool = True,
    simplify: bool = True,
    opset_version: int = 17,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export.
        output_path: Path for the output ONNX file.
        input_shape: Input tensor shape for tracing.
        config: Export configuration.
        dynamic_shapes: Whether to use dynamic batch dimension.
        simplify: Whether to simplify the ONNX model.
        opset_version: ONNX opset version.
        verbose: Whether to print detailed export info.

    Returns:
        Dictionary with export results and metadata.
    """
    if config is None:
        config = ExportConfig(
            opset_version=opset_version,
            simplify=simplify,
        )
        if dynamic_shapes:
            config.dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
        else:
            config.dynamic_axes = None

    start_time = time.time()
    results = {"success": False, "output_path": output_path}

    try:
        import torch

        model.eval()

        # Create dummy input
        dummy_input = torch.randn(*input_shape)

        # Move to same device as model
        device = next(model.parameters()).device
        dummy_input = dummy_input.to(device)

        # Export
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        if config.export_mode == "script":
            # Script-based export
            scripted_model = torch.jit.script(model)
            torch.onnx.export(
                scripted_model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=config.opset_version,
                do_constant_folding=config.do_constant_folding,
                input_names=config.input_names,
                output_names=config.output_names,
                dynamic_axes=config.dynamic_axes,
                verbose=verbose,
            )
        else:
            # Trace-based export
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=config.opset_version,
                do_constant_folding=config.do_constant_folding,
                input_names=config.input_names,
                output_names=config.output_names,
                dynamic_axes=config.dynamic_axes,
                verbose=verbose,
            )

        results["export_time"] = time.time() - start_time
        results["file_size_mb"] = os.path.getsize(output_path) / (1024 * 1024)

        # Simplify
        if config.simplify:
            simplify_result = simplify_onnx(output_path, output_path)
            results["simplified"] = simplify_result.get("success", False)
            if simplify_result.get("success"):
                results["file_size_mb"] = os.path.getsize(output_path) / (1024 * 1024)

        # Validate
        if config.check_model:
            check_result = check_onnx_model(output_path)
            results["validation"] = check_result

        # Test inference
        if config.test_export:
            test_result = test_onnx_inference(
                output_path, dummy_input.cpu().numpy(), model, device
            )
            results["inference_test"] = test_result

        results["success"] = True

    except ImportError:
        results["error"] = "PyTorch not available for ONNX export"
    except Exception as e:
        results["error"] = str(e)

    results["total_time"] = time.time() - start_time

    if results["success"]:
        print(f"[ONNX] Export successful: {output_path} ({results.get('file_size_mb', 0):.1f} MB)")
    else:
        print(f"[ONNX] Export failed: {results.get('error', 'Unknown error')}")

    return results


def simplify_onnx(input_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Simplify an ONNX model using onnx-simplifier.

    Args:
        input_path: Path to the ONNX model.
        output_path: Path for the simplified model (defaults to input).

    Returns:
        Dictionary with simplification results.
    """
    output_path = output_path or input_path
    result = {"success": False}

    try:
        import onnx
        from onnxsim import simplify

        model = onnx.load(input_path)
        simplified_model, check = simplify(model)

        if check:
            onnx.save(simplified_model, output_path)
            result["success"] = True
            result["original_size"] = os.path.getsize(input_path) / (1024 * 1024)
            result["simplified_size"] = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[ONNX] Simplified: {result['original_size']:.1f} MB -> {result['simplified_size']:.1f} MB")
        else:
            print("[ONNX] Simplification check failed")

    except ImportError:
        print("[ONNX] onnx-simplifier not available. Skipping simplification.")
    except Exception as e:
        result["error"] = str(e)
        print(f"[ONNX] Simplification failed: {e}")

    return result


def check_onnx_model(model_path: str) -> Dict[str, Any]:
    """
    Validate an ONNX model.

    Args:
        model_path: Path to the ONNX model.

    Returns:
        Dictionary with validation results.
    """
    result = {"valid": False}

    try:
        import onnx

        model = onnx.load(model_path)
        onnx.checker.check_model(model)

        result["valid"] = True
        result["opset_version"] = model.opset_import[0].version
        result["producer"] = model.producer_name
        result["ir_version"] = model.ir_version

        # Count nodes by type
        node_types = {}
        for node in model.graph.node:
            node_types[node.op_type] = node_types.get(node.op_type, 0) + 1
        result["node_counts"] = node_types
        result["total_nodes"] = len(model.graph.node)

        # Shape inference
        try:
            from onnx import shape_inference
            inferred_model = shape_inference.infer_shapes(model)
            result["shape_inference"] = "success"
        except Exception:
            result["shape_inference"] = "failed"

    except ImportError:
        result["error"] = "ONNX package not available"
    except Exception as e:
        result["error"] = str(e)

    return result


def test_onnx_inference(
    onnx_path: str,
    test_input: np.ndarray,
    pytorch_model: Any = None,
    device: str = "cpu",
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """
    Test ONNX model inference and compare with PyTorch output.

    Args:
        onnx_path: Path to the ONNX model.
        test_input: Test input numpy array.
        pytorch_model: Original PyTorch model for comparison.
        device: Device for PyTorch inference.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        Dictionary with test results.
    """
    result = {"success": False}

    try:
        import onnxruntime as ort

        # ONNX Runtime inference
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        onnx_output = session.run(None, {input_name: test_input.astype(np.float32)})[0]

        result["onnx_output_shape"] = onnx_output.shape
        result["onnx_output_range"] = (float(onnx_output.min()), float(onnx_output.max()))

        # Compare with PyTorch
        if pytorch_model is not None:
            import torch
            pytorch_model.eval()
            with torch.no_grad():
                torch_input = torch.from_numpy(test_input).to(device)
                torch_output = pytorch_model(torch_input).cpu().numpy()

            result["pytorch_output_shape"] = torch_output.shape

            # Compare outputs
            if torch_output.shape == onnx_output.shape:
                max_diff = float(np.max(np.abs(torch_output - onnx_output)))
                mean_diff = float(np.mean(np.abs(torch_output - onnx_output)))
                matches = np.allclose(torch_output, onnx_output, rtol=rtol, atol=atol)

                result["max_difference"] = max_diff
                result["mean_difference"] = mean_diff
                result["outputs_match"] = matches
            else:
                result["shape_mismatch"] = True

        # Measure latency
        latencies = []
        for _ in range(10):
            start = time.time()
            session.run(None, {input_name: test_input.astype(np.float32)})
            latencies.append(time.time() - start)

        result["avg_latency_ms"] = float(np.mean(latencies) * 1000)
        result["success"] = True

    except ImportError:
        result["error"] = "onnxruntime not available"
    except Exception as e:
        result["error"] = str(e)

    return result


def get_onnx_model_info(model_path: str) -> Dict[str, Any]:
    """
    Get information about an ONNX model file.

    Args:
        model_path: Path to the ONNX model.

    Returns:
        Dictionary of model information.
    """
    info = {"path": model_path}

    if not os.path.exists(model_path):
        info["exists"] = False
        return info

    info["exists"] = True
    info["file_size_mb"] = os.path.getsize(model_path) / (1024 * 1024)

    try:
        import onnx
        model = onnx.load(model_path)
        info["opset_version"] = model.opset_import[0].version
        info["ir_version"] = model.ir_version
        info["producer"] = model.producer_name

        # Input/output info
        inputs = []
        for inp in model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            inputs.append({"name": inp.name, "shape": shape})
        info["inputs"] = inputs

        outputs = []
        for out in model.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            outputs.append({"name": out.name, "shape": shape})
        info["outputs"] = outputs

    except ImportError:
        pass
    except Exception as e:
        info["error"] = str(e)

    return info
