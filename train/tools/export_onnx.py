from ultralytics import RTDETR
import sys

def export_onnx(model_path, output_path):
    model = RTDETR(model_path, task='detect')
    model.export(format='onnx', output=output_path)
    print(f"Model exported to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python export_onnx.py <model_path> <output_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_path = sys.argv[2]
    
    export_onnx(model_path, output_path)
    print("Export completed.")