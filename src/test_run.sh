export CUDA_VISIBLE_DEVICES=0

export IMAGE_HEIGHT=256
export IMAGE_WIDTH=256

export TEST_BATCH_SIZE=16

export MODEL_MEAN="(0.485, 0.456, 0.406)"
export MODEL_STD="(0.229, 0.224, 0.225)"

export BASE_MODEL="effectnet"

python test.py