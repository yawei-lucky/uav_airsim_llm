#!/bin/bash
module load cuda/11.8.0

# Change to project directory and execute evaluation script
# cd /home/users/ntu/yaweizha/scratch/TravelUAV

bash server_bkg.sh
sleep 8
bash scripts/eval.sh
