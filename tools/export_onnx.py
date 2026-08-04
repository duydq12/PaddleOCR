import onnx
import torch
print(torch.__version__)
import tools.program as program
from ppocr.modeling.architectures import build_model_torch


def main():
    onnx_path = "../weights/latin_1801/best_model.onnx"
    config["Architecture"]["Head"]["out_channels"] = 351
    model = build_model_torch(config["Architecture"])
    model.load_state_dict(torch.load("../weights/latin_1801/best_model.pth"))

    model.eval()

    dummy_input = torch.ones((1, 3, 32, 128), dtype=torch.float32)
    # with torch.no_grad():
    #     print(model(dummy_input))
    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      onnx_path,  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=14,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['inputs'],  # the model's input names
                      output_names=['outputs'],  # the model's output names
                      dynamic_axes={'inputs': {0: 'batch_size'},  # variable length axes
                                    'outputs': {0: 'batch_size'}})

    print(f"Model has been converted to ONNX and saved at {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")


if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess()
    main()
