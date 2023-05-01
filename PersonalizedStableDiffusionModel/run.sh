name=$1
#export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./data_example/$name"
export OUTPUT_DIR="./output_example/$name"

CUDA_VISIBLE_DEVICES=1 accelerate launch train_comcat_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of V* $name" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2.5e-3 \
  --lr_scheduler="linear" \
  --lr_warmup_steps=0 \
  --comcat_rank=2 \
  --max_train_steps=500


pip install diffusers==0.10.2
pip install transformers==4.25.1