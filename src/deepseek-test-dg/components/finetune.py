#!/bin/bash

# Set model, dataset, and output directories
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Change this to the model you want to fine-tune
DATASET_NAME="AI-MO/NuminaMath-TIR"  # Replace with your dataset
OUTPUT_DIR="/mnt/st1/results/DeepSeek-Train"
LOG_DIR="/mnt/st1/logs"
TASK="aime24"  # Set your custom task

# Training Parameters
LEARNING_RATE="2.0e-5"
NUM_TRAIN_EPOCHS=3
MAX_SEQ_LENGTH=4096
BATCH_SIZE=4
GRAD_ACCUMULATION_STEPS=4
GRAD_CHECKPOINTING="--gradient_checkpointing"
BF16="--bf16"
LOGGING_STEPS=5
EVAL_STEPS=100

# Slurm Parameters (for job submission)
SLURM_OUTPUT="/mnt/st1/logs/%x-%j.out"
SLURM_ERROR="/mnt/st1/logs/%x-%j.err"

# Model Fine-tuning using Accelerate with DeepSpeed (ZeRO-3)
def fine_tune_model() {
    echo "Starting Fine-Tuning with DeepSpeed (ZeRO-3) for $MODEL_NAME..."

    accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
        --model_name_or_path $MODEL_NAME \
        --dataset_name $DATASET_NAME \
        --learning_rate $LEARNING_RATE \
        --num_train_epochs $NUM_TRAIN_EPOCHS \
        --packing \
        --max_seq_length $MAX_SEQ_LENGTH \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUMULATION_STEPS \
        $GRAD_CHECKPOINTING \
        $BF16 \
        --logging_steps $LOGGING_STEPS \
        --eval_strategy steps \
        --eval_steps $EVAL_STEPS \
        --output_dir $OUTPUT_DIR
}

# Submit job using Slurm for SFT (Fine-tuning)
def submit_sft_slurm() {
    echo "Submitting Fine-Tuning job to Slurm..."

    sbatch --output=$SLURM_OUTPUT --err=$SLURM_ERROR slurm/sft.slurm \
        --model $MODEL_NAME --dataset $DATASET_NAME --accelerator zero3
}

# GRPO Training using Accelerate with DeepSpeed (ZeRO-3)
def grpo_training() {
    echo "Starting GRPO Training with DeepSpeed (ZeRO-3) for $MODEL_NAME..."

    accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
        --output_dir $OUTPUT_DIR/DeepSeek-R1-Distill-Qwen-7B-GRPO \
        --model_name_or_path $MODEL_NAME \
        --dataset_name $DATASET_NAME \
        --max_prompt_length 256 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --logging_steps 10 \
        $BF16
}

# Model Evaluation with lighteval
def evaluate_model() {
    echo "Evaluating Model $MODEL_NAME..."

    MODEL_ARGS="pretrained=$MODEL_NAME,dtype=float16,max_model_length=32768,gpu_memory_utilisation=0.8"
    OUTPUT_DIR=$OUTPUT_DIR/evals/$MODEL_NAME

    lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
        --custom-tasks src/open_r1/evaluate.py \
        --use-chat-template \
        --system-prompt="Please reason step by step, and put your final answer within \boxed{}." \
        --output-dir $OUTPUT_DIR
}

# Multi-GPU Evaluation with Data Parallelism
def evaluate_model_data_parallel() {
    NUM_GPUS=8
    echo "Evaluating Model with Data Parallelism across $NUM_GPUS GPUs..."

    MODEL_ARGS="pretrained=$MODEL_NAME,dtype=float16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
    OUTPUT_DIR=$OUTPUT_DIR/evals/$MODEL_NAME

    lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
        --custom-tasks src/open_r1/evaluate.py \
        --use-chat-template \
        --system-prompt="Please reason step by step, and put your final answer within \boxed{}." \
        --output-dir $OUTPUT_DIR
}

# Multi-GPU Evaluation with Tensor Parallelism
def evaluate_model_tensor_parallel() {
    NUM_GPUS=8
    echo "Evaluating Model with Tensor Parallelism across $NUM_GPUS GPUs..."

    MODEL_ARGS="pretrained=$MODEL_NAME,dtype=float16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
    OUTPUT_DIR=$OUTPUT_DIR/evals/$MODEL_NAME

    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
        --custom-tasks src/open_r1/evaluate.py \
        --use-chat-template \
        --system-prompt="Please reason step by step, and put your final answer within \boxed{}." \
        --output-dir $OUTPUT_DIR
}

# Main Execution
def main() {
    # Run Fine-Tuning (uncomment if you want to run fine-tuning)
    fine_tune_model
    
    # Submit Fine-Tuning job to Slurm (uncomment if you want to submit job)
    # submit_sft_slurm
    
    # Run GRPO Training (uncomment if you want to run GRPO)
    # grpo_training

    # Evaluate Model (Single GPU)
    evaluate_model
    
    # Evaluate Model with Data Parallelism (Multiple GPUs)
    # evaluate_model_data_parallel

    # Evaluate Model with Tensor Parallelism (Multiple GPUs)
    # evaluate_model_tensor_parallel
}

# Run the script
main
