# EpisodicLikeMemoryModel
Repository: Episodic Like Memory Model Code
Author: Jeff Krichmar
Email: jkrichma@uci.edu

Description:
------------
Python code to run simulation scenarios that created the results reported in, "Episodic-Like Memory in a Simulation of Cuttlefish Behavior", by S. Kandimalla, Q. Wong, K. Zheng and J. Krichmar, bioRxiv, 2025.
   More details on the scenarios can be found in this paper.

vlm_episodic_like.py demonstrates episodic-like memory by replicating cuttlefish behavioral experiments.

vlm_episodic_predprey.py demonstrates episodic-like memory by introducing two prey types and one predator. 
    If the WHAT_ACTIONS flag is true, the memory will be queried with what and when information. 
    Else, the memory will be queried with when and where information.

For both simulations, results will be saved in NPY files and figures will be saved in PNG files.

-------------
