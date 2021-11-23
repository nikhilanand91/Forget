import torch

def classify_batch(model_outputs, targets):
	classification = torch.zeros(len(model_outputs))
	for idx, output in enumerate(model_outputs):
		if torch.argmax(output) == targets[idx]:
			classification[idx] = 1
	return classification