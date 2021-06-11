from utils import load_model

model_reg = load_model("/home/med/Scrivania/models/rgcn/checkpoint-740000/", "pytorch_model.bin", True, "cpu")
model = load_model("/home/med/Scrivania/models/baseline/checkpoint-740000/", "pytorch_model.bin", False, "cpu")
