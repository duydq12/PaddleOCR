import numpy as np
import onnxruntime as ort

from ppocr.postprocess import ParseQLabelDecode

decode = ParseQLabelDecode(
    '/data/data-text-recognition/tr-resources/latin/vocab-latin.txt',
    True,
    max_text_length=32
)

images = np.load("/tmp/images.npy")

session = ort.InferenceSession("../weights/latin_1801/best_model.onnx")
# Get input details
inputs = session.get_inputs()
input_names = [input.name for input in inputs]

input_data = {input_names[0]: images}

outputs = session.run(None, input_data)

print(decode(outputs[0]))