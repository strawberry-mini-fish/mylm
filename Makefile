.PHONY: train train-mhc train-all clean

# Training script
PYTHON = ./.venv/bin/python
TRAIN_SCRIPT = ./cs336_basics/train.py

# Output directories
OUTPUT_DIR_ORIGINAL = ./checkpoints/original
OUTPUT_DIR_MHC = ./checkpoints/mhc

# Default target
.DEFAULT_GOAL := train

# ========== Shared hyperparameters (for fair comparison) ==========
VOCAB_SIZE = 50257
TOKENIZER = gpt2
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
BATCH_SIZE = 16
CONTEXT_LENGTH = 256
MAX_ITERS = 50000
LEARNING_RATE = 3e-4
WARMUP_ITERS = 2000
EVAL_INTERVAL = 1000
LOG_INTERVAL = 100
SAVE_INTERVAL = 5000

# ========== Training targets ==========

# Train original Transformer
train:
	@mkdir -p $(OUTPUT_DIR_ORIGINAL)
	@echo "Starting original Transformer training..."
	$(PYTHON) $(TRAIN_SCRIPT) \
		--model_type original \
		--vocab_size $(VOCAB_SIZE) \
		--tokenizer_name $(TOKENIZER) \
		--n_layer $(N_LAYER) \
		--n_head $(N_HEAD) \
		--n_embd $(N_EMBD) \
		--batch_size $(BATCH_SIZE) \
		--context_length $(CONTEXT_LENGTH) \
		--max_iters $(MAX_ITERS) \
		--learning_rate $(LEARNING_RATE) \
		--warmup_iters $(WARMUP_ITERS) \
		--eval_interval $(EVAL_INTERVAL) \
		--log_interval $(LOG_INTERVAL) \
		--save_interval $(SAVE_INTERVAL) \
		--output_dir $(OUTPUT_DIR_ORIGINAL)

# Train mHC Transformer
train-mhc:
	@mkdir -p $(OUTPUT_DIR_MHC)
	@echo "Starting mHC Transformer training..."
	$(PYTHON) $(TRAIN_SCRIPT) \
		--model_type mhc \
		--vocab_size $(VOCAB_SIZE) \
		--tokenizer_name $(TOKENIZER) \
		--n_layer $(N_LAYER) \
		--n_head $(N_HEAD) \
		--n_embd $(N_EMBD) \
		--batch_size $(BATCH_SIZE) \
		--context_length $(CONTEXT_LENGTH) \
		--max_iters $(MAX_ITERS) \
		--learning_rate $(LEARNING_RATE) \
		--warmup_iters $(WARMUP_ITERS) \
		--eval_interval $(EVAL_INTERVAL) \
		--log_interval $(LOG_INTERVAL) \
		--save_interval $(SAVE_INTERVAL) \
		--output_dir $(OUTPUT_DIR_MHC)

# Train both models for comparison
train-all: train train-mhc
	@echo "Both models trained successfully!"
	@echo "Original Transformer checkpoints: $(OUTPUT_DIR_ORIGINAL)"
	@echo "mHC Transformer checkpoints: $(OUTPUT_DIR_MHC)"

# ========== Cleanup ==========

clean:
	@read -p "Are you sure you want to delete all checkpoints? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf ./checkpoints; \
		echo "Cleanup complete"; \
	else \
		echo "Cleanup cancelled"; \
	fi
