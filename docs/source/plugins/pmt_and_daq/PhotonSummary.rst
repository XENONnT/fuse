=============
PhotonSummary
=============

Plugin Description
==================
fuse VerticalMergerPlugin that concatenates propagated photons for S1s, S2s and PMT afterpulses

Technical Details
-----------------

.. code-block:: python

   depends_on = ("propagated_s2_photons", "propagated_s1_photons", "pmt_afterpulses")
   provides = "photon_summary"
   data_kind = "propagated_photons"