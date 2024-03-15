# Micro Physics Simulation

This collection of plugins simulates the signal generation processes in LXe. This part of the simulation was formerly handled by epix.

The relevant processes are:
- Reading and chunking root files. The event times are set in this step based on a given event rate.
- Microclustering of interactions is performed in two steps. First the clusters are determined and then merged and classified in a second step.
- Electric field assignemt: Each interaction is assigned an electric field based on the position of the interaction.
- Quanta generation: The number of photons and electrons is determined for each interaction.
