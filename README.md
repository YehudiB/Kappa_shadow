# xdsmask


## Pipeline

1. Visualizer webpage allows you to write out a theoretical mask as
   .npz file
2. The mask should then be **refined** to match reality. First an
   angular offset should be found that best matches reality.
   xds_shadow_refine.py tune-offset
3. The mask can then be adapted by fitting the transition of low to high 
   background along the edge of the theoretical mask.
   python xds_shadow_refine.py refine
   


## Install


## Usage

