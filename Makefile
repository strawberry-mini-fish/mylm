.PHONY: tokenize train train-mhc train-all train-compare clean clean-cache

# Training script
PYTHON = ./.venv/bin/python
TRAIN_SCRIPT = ./cs336_basics/train.py
TOKENIZE_SCRIPT = ./cs336_basics/pretokenize.py

# Output directories
OUTPUT_DIR_ORIGINAL = ./checkpoints/original
OUTPUT_DIR_MHC = ./checkpoints/mhc
TOKENIZED_DATA_DIR = ./tokenized_data

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

# mHC specific hyperparameters
MHC_LEARNING_RATE = 1e-4
MHC_WARMUP_ITERS = 4000
MHC_EXPANSION_RATE = 2

# Tokenization settings
MAX_SAMPLES = 100000

# ========== Tokenization ==========

# Pre-tokenize dataset (run this before training for faster startup)
tokenize:
	@echo "Pre-tokenizing dataset..."
	$(PYTHON) $(TOKENIZE_SCRIPT) \
		--dataset roneneldan/TinyStories \
		--tokenizer $(TOKENIZER) \
		--context_length $(CONTEXT_LENGTH) \
		--output_dir $(TOKENIZED_DATA_DIR) \
		--splits train validation \
		--max_samples $(MAX_SAMPLES)
	@echo "Tokenization complete! Cached in $(TOKENIZED_DATA_DIR)"

# Tokenize full dataset (no limit)
tokenize-full:
	@echo "Pre-tokenizing FULL dataset (this may take a while)..."
	$(PYTHON) $(TOKENIZE_SCRIPT) \
		--dataset roneneldan/TinyStories \
		--tokenizer $(TOKENIZER) \
		--context_length $(CONTEXT_LENGTH) \
		--output_dir $(TOKENIZED_DATA_DIR) \
		--splits train validation
	@echo "Tokenization complete! Cached in $(TOKENIZED_DATA_DIR)"

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
		--output_dir $(OUTPUT_DIR_ORIGINAL) \
		--max_samples $(MAX_SAMPLES)

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
		--learning_rate $(MHC_LEARNING_RATE) \
		--warmup_iters $(MHC_WARMUP_ITERS) \
		--eval_interval $(EVAL_INTERVAL) \
		--log_interval $(LOG_INTERVAL) \
		--save_interval $(SAVE_INTERVAL) \
		--output_dir $(OUTPUT_DIR_MHC) \
		--max_samples $(MAX_SAMPLES) \
		--expansion_rate $(MHC_EXPANSION_RATE) \
		--grad_clip 1.0

# Train both models for comparison
train-all: train train-mhc
	@echo "Both models trained successfully!"
	@echo "Original Transformer checkpoints: $(OUTPUT_DIR_ORIGINAL)"
	@echo "mHC Transformer checkpoints: $(OUTPUT_DIR_MHC)"

# Train both models sequentially with TensorBoard logging for comparison
train-compare:
	@echo "Training both models with TensorBoard logging..."
	$(PYTHON) ./train_compare.py \
		--vocab_size $(VOCAB_SIZE) \
		--tokenizer_name $(TOKENIZER) \
		--n_layer $(N_LAYER) \
		--n_head $(N_HEAD) \
		--n_embd $(N_EMBD) \
		--batch_size $(BATCH_SIZE) \
		--context_length $(CONTEXT_LENGTH) \
		--max_iters $(MAX_ITERS) \
		--learning_rate $(LEARNING_RATE) \
		--mhc_learning_rate $(MHC_LEARNING_RATE) \
		--warmup_iters $(MHC_WARMUP_ITERS) \
		--eval_interval $(EVAL_INTERVAL) \
		--log_interval $(LOG_INTERVAL) \
		--expansion_rate $(MHC_EXPANSION_RATE) \
		--max_samples $(MAX_SAMPLES) \
		--output_dir ./checkpoints \
		--tensorboard_dir ./runs/compare
	@echo "Training complete! View TensorBoard: tensorboard --logdir ./runs/compare"

# ========== Cleanup ==========

# Clean checkpoints
clean:
	@read -p "Are you sure you want to delete all checkpoints? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf ./checkpoints; \
		echo "Cleanup complete"; \
	else \
		echo "Cleanup cancelled"; \
	fi

# Clean tokenized data cache
clean-cache:
	@read -p "Are you sure you want to delete all tokenized data cache? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf $(TOKENIZED_DATA_DIR); \
		echo "Cache cleanup complete"; \
	else \
		echo "Cache cleanup cancelled"; \
	fi
