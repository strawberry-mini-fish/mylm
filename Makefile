.PHONY: train clean

# 训练脚本
PYTHON = ./.venv/bin/python
TRAIN_SCRIPT = ./cs336_basics/train.py
OUTPUT_DIR = ./checkpoints

# 默认目标
.DEFAULT_GOAL := train

# 标准训练配置
train:
	@mkdir -p $(OUTPUT_DIR)
	@echo "开始标准训练..."
	$(PYTHON) $(TRAIN_SCRIPT) \
		--vocab_size 50257 \
		--tokenizer_name gpt2 \
		--n_layer 6 \
		--n_head 6 \
		--n_embd 384 \
		--batch_size 16 \
		--context_length 256 \
		--max_iters 50000 \
		--learning_rate 3e-4 \
		--warmup_iters 2000 \
		--eval_interval 1000 \
		--log_interval 100 \
		--save_interval 5000 \
		--output_dir $(OUTPUT_DIR)

# 清理
clean:
	@read -p "确定要删除检查点吗？ [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf $(OUTPUT_DIR); \
		echo "清理完成"; \
	else \
		echo "取消清理"; \
	fi