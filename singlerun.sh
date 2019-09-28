
#!/bin/bash

episodes=3500


python src/multirun.py -r0 -c0 -e$episodes
python src/multirun.py -r1 -c1 -e$episodes

