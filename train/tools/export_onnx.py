from ultralytics import RTDETR
import sys

def export_onnx(model_path):
    model = RTDETR(model_path)
    model.export(format='onnx')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python export_onnx.py <model_path> ")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    export_onnx(model_path)
    print("Export completed.")