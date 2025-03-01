PYTHON = python3

# Model files
MODEL_PTH = symbol_classifier.pth
MODEL_MLPACKAGE = symbol_classifier.mlpackage  # This is a directory

# Targets
.PHONY: all train convert clean

all: train convert

train:
	rm -f $(MODEL_PTH)  # Delete old model before training
	$(PYTHON) train.py
	@echo "Training completed and new model saved as $(MODEL_PTH)."

convert:
	rm -rf $(MODEL_MLPACKAGE)  # Delete old converted model (directory)
	$(PYTHON) convert.py
	@echo "Model converted to Core ML and saved as $(MODEL_MLPACKAGE)."

clean:
	rm -f $(MODEL_PTH)  # Remove model file
	rm -rf $(MODEL_MLPACKAGE)  # Remove Core ML model package (directory)
	@echo "Cleaned up model files."
