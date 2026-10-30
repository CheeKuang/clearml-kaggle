
'''
ONNX weight shrinking for the model stage.

Supports:
  - integer_weights: store large float constants as int8 + Cast/Mul/Add dequant
  - float_weights: round large float constants to a limited number of float levels
  - print_weight_info: inspect weight sizes before/after shrink

Notebook-oriented API: shrink_onnx_weights(...).

Copied from chisel-pipeline-convert (libs/train_validate_deploy/shrink.py); the only change is the `del ...[:]` below.
Command line:  python -m src.shrink models/road.onnx models/road.shrunk.onnx [--method float_weights] [--ir-version 8]
'''

#--- IMPORTS ---#

import os

import numpy as np
import onnx
import onnx_graphsurgeon as gs


# Don't quantise constants smaller than this.
DEFAULT_MIN_ELEMENTS = 16 * 1024

#--- SHRINK ---#

def replace_tensor_for_subgraph(graph, original_tensor_name, new_tensor):
    """Replace a tensor in a graph with a new tensor.

    Args:
        graph: The graph to modify.
        original_tensor_name: The name of the tensor to replace.
        new_tensor: The tensor to replace it with.
    """
    for node in graph.nodes:
        for subgraph in node.attrs.values():
            if isinstance(subgraph, gs.Graph):
                replace_tensor_for_subgraph(subgraph, original_tensor_name, new_tensor)
        for i, tensor in enumerate(node.inputs):
            if tensor.name == original_tensor_name:
                node.inputs[i] = new_tensor

    for i, tensor in enumerate(graph.outputs):
        if tensor.name == original_tensor_name:
            graph.outputs[i] = new_tensor


def gather_initializers_in_graph(graph, all_initializers):
    for initializer in graph.initializer:
        all_initializers[initializer.name] = initializer

    del graph.initializer[:]      # protobuf 5 (our pin) has no .clear() on repeated fields

    for node in graph.node:
        if node.op_type == "If":
            for attr in node.attribute:
                if attr.name == "then_branch":
                    all_initializers = gather_initializers_in_graph(
                        attr.g, all_initializers
                    )
                elif attr.name == "else_branch":
                    all_initializers = gather_initializers_in_graph(
                        attr.g, all_initializers
                    )

    return all_initializers


def hoist_subgraph_initializers(onnx_model):
    """GraphSurgeon seems to leave duplicated initializers in the graph, so remove them."""

    all_initializers = {}
    gather_initializers_in_graph(onnx_model.graph, all_initializers)

    for _, initializer in all_initializers.items():
        onnx_model.graph.initializer.append(initializer)

    return onnx_model


def quantize_tensor(name, value_tensor, original_output_tensor_name, root_graph):
    """Quantize a constant tensor to int8, reconstructed via Cast/Mul/Add.

    Args:
        name: The name of the tensor to quantize.
        value_tensor: The tensor to quantize.
        original_output_tensor_name: The name of the original tensor in the graph.
        root_graph: The root graph of the model.
    """
    dtype = value_tensor.dtype
    np_dtype = onnx.helper.np_dtype_to_tensor_dtype(dtype)
    float_values = value_tensor.values
    min_val = np.min(float_values)
    max_val = np.max(float_values)
    range_val = max_val - min_val
    inverse_range = 1.0 / range_val
    zero_point = round(-min_val * inverse_range * 255.0) - 128
    quantized_values = np.round(float_values * inverse_range * 255.0) + zero_point
    quantized_values = np.clip(quantized_values, -128, 127).astype(np.int8)

    quantized_tensor = gs.Constant(name=f"{name}_quantized", values=quantized_values)

    scale_value = range_val / 255.0
    scale_tensor = gs.Constant(
        name=f"{name}_scale", values=np.array([scale_value], dtype=dtype)
    )

    zero_point_tensor = gs.Constant(
        name=f"{name}_zero_point",
        values=np.array([-zero_point * scale_value], dtype=dtype),
    )

    # DequantizeLinear is surprisingly slow in OnnxRuntime, so achieve the
    # same effect with a Cast, Mul, and Add.
    cast_tensor = gs.Variable(
        name=f"{name}_cast_tensor", dtype=dtype, shape=value_tensor.shape
    )
    cast_node = gs.Node(
        op="Cast",
        name=f"{name}_cast_node",
        inputs=[quantized_tensor],
        outputs=[cast_tensor],
        attrs={"to": np_dtype},
    )

    mul_tensor = gs.Variable(
        name=f"{name}_mul_tensor", dtype=dtype, shape=value_tensor.shape
    )
    mul_node = gs.Node(
        op="Mul",
        name=f"{name}_mul_node",
        inputs=[cast_tensor, scale_tensor],
        outputs=[mul_tensor],
    )

    add_tensor = gs.Variable(
        name=f"{name}_add_tensor", dtype=dtype, shape=value_tensor.shape
    )
    add_node = gs.Node(
        op="Add",
        name=f"{name}_add_node",
        inputs=[mul_tensor, zero_point_tensor],
        outputs=[add_tensor],
    )

    replace_tensor_for_subgraph(root_graph, original_output_tensor_name, add_tensor)

    root_graph.nodes.append(cast_node)
    root_graph.nodes.append(mul_node)
    root_graph.nodes.append(add_node)


def float_quantize_node(
    name, value_tensor, original_output_tensor_name, root_graph, levels=256
):
    """Quantize a constant tensor to a small number of float values.

    Args:
        name: The name of the tensor to quantize.
        value_tensor: The tensor to quantize.
        original_output_tensor_name: The name of the original tensor in the graph.
        root_graph: The root graph of the model.
        levels: The number of levels to quantize to.
    """
    dtype = value_tensor.dtype
    float_values = value_tensor.values
    min_val = np.min(float_values)
    max_val = np.max(float_values)
    range_val = max_val - min_val
    inverse_range = 1.0 / range_val
    half_levels = levels / 2
    zero_point = round(-min_val * inverse_range * (levels - 1)) - half_levels
    scale_value = range_val / (levels - 1)
    quantized_values = (
        np.round(float_values * inverse_range * (levels - 1)) + zero_point
    )
    quantized_values = np.clip(quantized_values, -half_levels, (half_levels - 1))
    dequantized_values = (
        (quantized_values.astype(np.int32) - zero_point) * scale_value
    ).astype(dtype)

    dequantized_tensor = gs.Constant(
        name=f"{name}_dequantized", values=dequantized_values
    )

    replace_tensor_for_subgraph(
        root_graph, original_output_tensor_name, dequantized_tensor
    )


def quantize_weights_for_graph(
    graph,
    root_graph,
    already_processed,
    min_elements=DEFAULT_MIN_ELEMENTS,
    float_quantization=False,
    float_levels=256,
    verbose=False,
):
    for node in graph.nodes:
        for subgraph in node.attrs.values():
            if isinstance(subgraph, gs.Graph):
                if verbose:
                    print(f"Processing subgraph {subgraph.name}")
                already_processed = quantize_weights_for_graph(
                    subgraph,
                    root_graph,
                    already_processed,
                    min_elements,
                    float_quantization,
                    float_levels,
                    verbose,
                )
        if node.op != "Constant":
            continue
        name = node.name
        value_tensor = node.attrs["value"]
        if value_tensor.dtype not in [np.float16, np.float32, np.float64]:
            continue
        original_output_tensor_name = node.outputs[0].name
        if original_output_tensor_name in already_processed:
            continue
        already_processed.add(original_output_tensor_name)
        elements = np.prod(value_tensor.shape)
        if elements < min_elements:
            continue
        if verbose:
            print(f"Processing node {name}")
        if float_quantization:
            float_quantize_node(
                name,
                value_tensor,
                original_output_tensor_name,
                root_graph,
                levels=float_levels,
            )
        else:
            quantize_tensor(
                name, value_tensor, original_output_tensor_name, root_graph
            )

    for name, value_tensor in graph.tensors().items():
        if value_tensor.dtype not in [np.float16, np.float32, np.float64]:
            continue
        if value_tensor.__class__ != gs.Constant:
            continue
        original_output_tensor_name = name
        if original_output_tensor_name in already_processed:
            continue
        already_processed.add(original_output_tensor_name)
        elements = np.prod(value_tensor.shape)
        if elements < min_elements:
            continue
        if verbose:
            print(f"Processing initializer {name}")
        if float_quantization:
            float_quantize_node(
                name,
                value_tensor,
                original_output_tensor_name,
                root_graph,
                levels=float_levels,
            )
        else:
            quantize_tensor(
                name, value_tensor, original_output_tensor_name, root_graph
            )

    return already_processed


def quantize_weights(
    input_data,
    min_elements=DEFAULT_MIN_ELEMENTS,
    float_quantization=False,
    float_levels=256,
    verbose=False,
    ir_version=None,
):
    """Quantize the weights of an ONNX model.

    Args:
        input_data: The path or contents of the ONNX model to quantize.
        min_elements: The minimum number of elements a tensor must have to be quantized.
        float_quantization: If True, store the quantized values as float, not integers.
        float_levels: The number of levels to quantize to if using float quantization.
        verbose: If True, log detailed information about the weight processing.
        ir_version: The IR version to use for the output ONNX files.
    """
    if verbose:
        print(
            f"quantize_weights(input_data, min_elements={min_elements}, "
            f"float_quantization={float_quantization}, float_levels={float_levels})"
        )

    graph = gs.import_onnx(input_data)

    already_processed = set()
    quantize_weights_for_graph(
        graph,
        graph,
        already_processed,
        min_elements,
        float_quantization,
        float_levels,
        verbose,
    )

    graph.cleanup(remove_unused_graph_inputs=False).toposort(recurse_subgraphs=True)

    no_shape_model = gs.export_onnx(graph)
    deduped_model = hoist_subgraph_initializers(no_shape_model)
    new_model = onnx.shape_inference.infer_shapes(deduped_model)
    if ir_version is not None:
        new_model.ir_version = ir_version

    onnx.checker.check_model(new_model)

    return new_model


def print_weight_info_for_graph(
    onnx_graph,
    total_bytes,
    node_count,
    initializer_count,
    already_processed,
    min_elements=DEFAULT_MIN_ELEMENTS,
):
    for node in onnx_graph.node:
        value_tensor = None
        for attribute in node.attribute:
            if attribute.name == "value":
                value_tensor = attribute.t
            if attribute.HasField("g"):
                subgraph = attribute.g
                total_bytes, node_count, initializer_count, already_processed = (
                    print_weight_info_for_graph(
                        subgraph,
                        total_bytes,
                        node_count,
                        initializer_count,
                        already_processed,
                        min_elements,
                    )
                )
        if node.op_type != "Constant":
            continue
        output_tensor_name = node.output[0]
        if output_tensor_name in already_processed:
            continue
        already_processed.add(output_tensor_name)
        name = node.name
        elements = np.prod(value_tensor.dims)
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(value_tensor.data_type)
        byte_count = int(elements * np_dtype.itemsize)
        total_bytes += byte_count
        if elements < min_elements:
            continue
        node_count += 1
        print(
            f"Node: {name}: {value_tensor.dims} - {elements} elements, {np_dtype}, {byte_count:,} bytes"
        )

    duplicate_names = set()
    for value_tensor in onnx_graph.initializer:
        name = value_tensor.name
        if name in already_processed:
            duplicate_names.add(name)
            continue
        already_processed.add(name)
        elements = np.prod(value_tensor.dims)
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(value_tensor.data_type)
        byte_count = int(elements * np_dtype.itemsize)
        total_bytes += byte_count
        if elements < min_elements:
            continue
        initializer_count += 1
        print(
            f"Initializer: {name}: {value_tensor.dims} - {elements:,} elements, {np_dtype}, {byte_count:,} bytes"
        )

    if len(duplicate_names) > 0:
        print(f"Duplicate initializers: {duplicate_names}")

    return total_bytes, node_count, initializer_count, already_processed


def print_weight_info(filename_or_model, min_elements=DEFAULT_MIN_ELEMENTS):
    """Print information about the size of the weights in an ONNX model.

    Args:
        filename_or_model: Path to an ONNX file, or an in-memory ONNX model.
        min_elements: Minimum tensor size to include in the detailed listing.
    """
    if isinstance(filename_or_model, str):
        filename = filename_or_model
        onnx_model = onnx.load(filename)
        file_byte_count = os.path.getsize(filename)
        print(f"Model: {filename}")
    else:
        onnx_model = filename_or_model
        file_byte_count = onnx_model.ByteSize()

    total_bytes = 0
    node_count = 0
    initializer_count = 0
    already_processed = set()

    total_bytes, node_count, initializer_count, already_processed = (
        print_weight_info_for_graph(
            onnx_model.graph,
            total_bytes,
            node_count,
            initializer_count,
            already_processed,
            min_elements,
        )
    )

    print(f"Total nodes: {node_count}")
    print(f"Total initializers: {initializer_count}")
    print(
        f"Total bytes from weights: {total_bytes:,} bytes, "
        f"{file_byte_count - total_bytes:,} bytes from other data"
    )
    print("-------------------------------------------")


def shrink_onnx_weights(
    input_path,
    output_path,
    method="integer_weights",
    ir_version=None,
    float_levels=256,
    min_elements=DEFAULT_MIN_ELEMENTS,
    verbose=False,
    print_info=False,
    external_data=False,
):
    """Thin notebook entrypoint: load → optional info → shrink → save → optional info.

    Args:
        input_path: Path to the input ONNX model.
        output_path: Path to write the shrunk ONNX model.
        method: "integer_weights" (default) or "float_weights".
        ir_version: Optional ONNX IR version to set on the output model.
        float_levels: Levels used when method is "float_weights".
        min_elements: Minimum tensor size to shrink.
        verbose: Log detailed weight-processing information.
        print_info: If True, print weight info before and after shrinking.
        external_data: If True, save with external data.

    Returns:
        dict with input_path, output_path, method, bytes_before, bytes_after.
    """
    if method not in ("integer_weights", "float_weights"):
        raise ValueError(
            f"Unsupported method '{method}'. Use 'integer_weights' or 'float_weights'."
        )

    bytes_before = os.path.getsize(input_path)
    if print_info:
        print_weight_info(input_path, min_elements=min_elements)

    original_model = onnx.load(input_path)
    new_model = quantize_weights(
        original_model,
        min_elements=min_elements,
        float_quantization=(method == "float_weights"),
        float_levels=float_levels,
        verbose=verbose,
        ir_version=ir_version,
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    onnx.save(new_model, output_path, save_as_external_data=external_data)
    bytes_after = os.path.getsize(output_path)

    print(f'Successfully wrote compressed model at {output_path}.')

    if print_info:
        print_weight_info(output_path, min_elements=min_elements)

    if verbose:
        print(
            f"Original file size: {bytes_before:,} bytes, "
            f"shrunk file size: {bytes_after:,} bytes"
        )

    return {
        "input_path": input_path,
        "output_path": output_path,
        "method": method,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Shrink the weights of an ONNX model (int8 storage by default)")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--method", default="integer_weights", choices=["integer_weights", "float_weights"])
    ap.add_argument("--ir-version", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    r = shrink_onnx_weights(a.input, a.output, method=a.method, ir_version=a.ir_version, verbose=a.verbose)
    print(f"{r['bytes_before'] / 1e6:.1f} MB -> {r['bytes_after'] / 1e6:.1f} MB")
