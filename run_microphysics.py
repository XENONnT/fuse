import strax

from micro_physics.input import input_plugin 
from micro_physics.find_cluster import find_cluster 
from micro_physics.merge_cluster import cluster_merging
from micro_physics.electric_field import electic_field
from micro_physics.yields import nest_yields, bbf_yields
from micro_physics.output import output_plugin


st = strax.Context(register = [input_plugin,
                               find_cluster,
                               cluster_merging,
                               electic_field,
                               nest_yields,
                               #bbf_yields,
                               output_plugin],
                   storage = [strax.DataDirectory('./epix_data')]
                  )

st.set_config({"path": "/project2/lgrandi/xenonnt/simulations/testing",
               "file_name": "pmt_neutrons_100.root",
               "ChunkSize": 50
              })


geant4_interactions = st.make("00000","geant4_interactions")
cluster_index = st.make("00000","cluster_index")
clustered_interactions = st.make("00000","clustered_interactions")
electic_field_values = st.make("00000",[ "electic_field_values"])
quanta = st.make("00000",[ "quanta"])
wfsim_instructions = st.make("00000",[ "wfsim_instructions"])