# WFSS Background

This preforms the master sky subtraction of a wide-field slitless spectroscopic (WFSS) image from JWST NIRCam and NIRISS while making corrections for the $1/f$ noise.

## The Model

The WFSS image is assumed to be decomposed into 4 distinct images:

1. the scene of dispersed sources;
2. the astrophysical background;
3. the fast-read 1/f pattern noise; and  
4. the slow-read 1/f pattern noise.

The scene of the dispersed sources is generally the item of interest to most observers, and requires very careful modeling and extraction techniques to fully understand.  Since the present goal is related to the background, the dispersed scene is taken only as a mask, where source