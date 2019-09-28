
#!/bin/bash

run=0
context=$((run%2))
episodes=2
numberofswitches=100
folder="/run/media/bernardo/481EDFB11EDF95F2/multiruns/"

python src/multirun.py -r$run -c$context -e$episodes -f$folder

prevrun=$((run+0))
run=$((run+1))
context=$((run%2))
python src/multirun.py -r$run -c$context -e$episodes -m$prevrun -f$folder


for ((i=0;i<=numberofswitches;i++)); do
    prevprevrun=$((prevrun+0))
    prevrun=$((run+0))
    run=$((run+1))
    context=$((run%2))
    python src/multirun.py -r$run -c$context -e$episodes -m$prevrun -o$prevprevrun -f$folder
    rm $folder*$prevprevrun.h5_Memory.pickle
done

