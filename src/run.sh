export CUDA_VISIBLE_DEVICES=0

export IMAGE_HEIGHT=256
export IMAGE_WIDTH=256
export EPOCHS=100

export TRAIN_BATCH_SIZE=64
export TEST_BATCH_SIZE=32

export MODEL_MEAN="(0.485, 0.456, 0.406)"
export MODEL_STD="(0.229, 0.224, 0.225)"


export BASE_MODEL="resnet34"

export TRAINING_FOLDS="(0, 1)"
export VALIDATION_FOLDS="(2,)"
python train.py


export TRAINING_FOLDS="(0, 2)"
export VALIDATION_FOLDS="(1,)"
python train.py



export TRAINING_FOLDS="(1,2)"
export VALIDATION_FOLDS="(0,)"
python train.py
