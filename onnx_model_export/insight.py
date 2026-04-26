import os
import onnx
import onnxruntime

def find_onnx_files(current_dir):
    onnx_files = []
    for filename in os.listdir(current_dir):
        if filename.lower().endswith('.onnx') and os.path.isfile(os.path.join(current_dir, filename)):
            onnx_files.append(os.path.join(current_dir, filename))
    return onnx_files

def get_model_info(model_path):
    try:
        runtime_model = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_shape = runtime_model.get_inputs()[0].shape
        output_shape = runtime_model.get_outputs()[0].shape
        model = onnx.load(model_path)
        opset_version = model.opset_import[0].version
        return [os.path.basename(model_path), str(input_shape), str(output_shape), str(opset_version)]
    except Exception as e:
        return [os.path.basename(model_path), "Error", "Error", str(e)]

def print_table(rows, headers):
    # Calculate column widths
    cols = list(zip(*([headers] + rows)))
    col_widths = [max(len(str(item)) for item in col) for col in cols]
    # Print header
    header_row = " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-+-".join('-' * w for w in col_widths))
    # Print rows
    for row in rows:
        print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_files = find_onnx_files(current_dir)
    headers = ["File Name", "Input Shape", "Output Shape", "Opset Version"]
    rows = [get_model_info(file) for file in onnx_files]
    print_table(rows, headers)

if __name__ == "__main__":
    main()