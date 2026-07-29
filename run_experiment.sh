#!/bin/bash

EPOCHS=20
THRESHOLDS=(0.5 1.0 1.5)

run_experiment() {
local dataset=$1
local arch=$2
local time=$3
local batch=$4
local vth=$5

python main.py \
    --dataset $dataset \
    --architecture $arch \
    --epochs $EPOCHS \
    --batch_size $batch \
    --Time $time \
    --v_threshold $vth \
    --use_wandb

sleep 5


}

for t in 2 4 8 16 32; do
for vth in "${THRESHOLDS[@]}"; do
run_experiment "nmnist" "SpikingMLP" $t 64 $vth
done
done

for t in 2 4 8 16 32; do
for vth in "${THRESHOLDS[@]}"; do
run_experiment "cifar10" "SpikingVGG4" $t 16 $vth
done
done

for t in 2 4 8 16 32; do
for vth in "${THRESHOLDS[@]}"; do
run_experiment "dvs_gesture" "SpikingVGG5" $t 16 $vth
done
done

for t in 2 4 8; do
for vth in "${THRESHOLDS[@]}"; do
run_experiment "nepic_kitchens" "SpikingVGG8" $t 4 $vth
done
done

for t in 2 4 8; do
for vth in "${THRESHOLDS[@]}"; do
run_experiment "nepic_kitchens" "SpikingResNet18" $t 4 $vth
done
done