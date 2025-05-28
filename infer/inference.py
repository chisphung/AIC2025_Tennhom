from ultralytics import RTDETR
import sys
def infer_onnx(model_path, image_path):
    model = RTDETR(model_path, task='detect')
    results = model(image_path, conf=0.45)
    return results

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python infer_onnx.py <model_path> <image_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    results = infer_onnx(model_path, image_path)
    print(results)
    # Save results to a file or process them as needed
    with open('results.txt', 'w') as f:
        for result in results:
            f.write(str(result) + '\n')
    print("Inference completed. Results saved to results.txt")
