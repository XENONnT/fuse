import strax
import cutax

from micro_physics.input import input_plugin 
from micro_physics.find_cluster import find_cluster 
from micro_physics.merge_cluster import cluster_merging
from micro_physics.electric_field import ElectricField
from micro_physics.yields import nest_yields, bbf_yields
from micro_physics.output import output_plugin

from detector_physics.S1_scintillation_and_photon_propagation import S1_scintillation_and_propagation
from detector_physics.electron_drift import electron_drift
from detector_physics.electron_extraction import electron_extraction
from detector_physics.electron_timing import electron_timing
from detector_physics.secondary_scintillation import scintillation
from detector_physics.S2_photon_propagation import S2_photon_distributions_and_timing


st = cutax.contexts.xenonnt_sim_SR0v3_cmt_v9()
st.register(input_plugin)
st.register(find_cluster)
st.register(cluster_merging)
st.register(ElectricField)
st.register(nest_yields)
st.register(output_plugin)

st.register(S1_scintillation_and_propagation)

st.register(electron_drift)
st.register(electron_extraction)
st.register(electron_timing)
st.register(scintillation)
st.register(S2_photon_distributions_and_timing)


st.set_config({"path": "/project2/lgrandi/xenonnt/simulations/testing",
               "file_name": "pmt_neutrons_100.root",
               "ChunkSize": 50,
              })

#Microphysics (former epix)
st.make("00000","geant4_interactions")
st.make("00000","cluster_index")
st.make("00000","clustered_interactions")
st.make("00000",[ "electic_field_values"])
st.make("00000",[ "quanta"])
st.make("00000",[ "wfsim_instructions"])

#S1 detector physics (former WFsim)
st.make("00000",[ "S1_channel_and_timings"])

#S2 detector physics (former WFsim)
st.make("00000", "drifted_electrons")
st.make("00000", "extracted_electrons")
st.make("00000", "electron_time")
st.make("00000", "photons")
st.make("00000", "photon_channels_and_timeing")