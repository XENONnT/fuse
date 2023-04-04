import strax
import numpy as np
import wfsim
import awkward as ak

import epix
from epix.common import offset_range

class output_plugin(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ["clustered_interactions", "quanta", "electric_field_values"] #Add times later
    
    provides = "wfsim_instructions"
    data_kind = 'wfsim_instructions'
    
    dtype =         [(('Waveform simulator event number.', 'event_number'), np.int32),
                     (('Quanta type (S1 photons or S2 electrons)', 'type'), np.int8),
                     (('Time of the interaction [ns]', 'time'), np.int64),
                     (('End Time of the interaction [ns]', 'endtime'), np.int64),
                     (('X position of the cluster [cm]', 'x'), np.float32),
                     (('Y position of the cluster [cm]', 'y'), np.float32),
                     (('Z position of the cluster [cm]', 'z'), np.float32),
                     (('Number of quanta', 'amp'), np.int32),
                     (('Recoil type of interaction.', 'recoil'), np.int8),
                     (('Energy deposit of interaction', 'e_dep'), np.float32),
                     (('Eventid like in geant4 output rootfile', 'g4id'), np.int32),
                     (('Volume id giving the detector subvolume', 'vol_id'), np.int32),
                     (('Local field [ V / cm ]', 'local_field'), np.float64),
                     (('Number of excitons', 'n_excitons'), np.int32),
                     (('X position of the primary particle [cm]', 'x_pri'), np.float32),
                     (('Y position of the primary particle [cm]', 'y_pri'), np.float32),
                     (('Z position of the primary particle [cm]', 'z_pri'), np.float32),
                    ]

    def compute(self, clustered_interactions):
        
        instructions = self.awkward_to_wfsim_row_style(clustered_interactions)
        
        return instructions
    
    
    def awkward_to_wfsim_row_style(self, interactions):
        """
        Converts awkward array instructions into instructions required by
        WFSim.
        :param interactions: awkward.Array containing GEANT4 simulation
            information.
        :return: Structured numpy.array. Each row represents either a S1 or
            S2
        """
        if len(interactions) == 0:
            return np.empty(0, dtype=wfsim.instruction_dtype)

        ninteractions = len(interactions['ed'])
        res = np.zeros(2 * ninteractions, dtype=self.dtype)

        # TODO: Currently not supported rows with only electrons or photons due to
        # this super odd shape
        for i in range(2):
            
            structure = interactions["structure"][interactions["structure"]>=0]
            evtid = epix.reshape_awkward(interactions["evtid"], structure)
            
            res['event_number'][i::2] = offset_range(ak.to_numpy(ak.num(evtid)))
            res['type'][i::2] = i + 1
            res['x'][i::2] = interactions['x']
            res['y'][i::2] = interactions['y']
            res['z'][i::2] = interactions['z']
            res['x_pri'][i::2] = interactions['x_pri']
            res['y_pri'][i::2] = interactions['y_pri']
            res['z_pri'][i::2] = interactions['z_pri']
            res['g4id'][i::2] = interactions['evtid']
            res['vol_id'][i::2] = interactions['vol_id']
            res['e_dep'][i::2] = interactions['ed']
            if 'local_field' in res.dtype.names:
                res['local_field'][i::2] = interactions['e_field']

            recoil = interactions['nestid']
            res['recoil'][i::2] = np.where(np.isin(recoil, [0, 6, 7, 8, 11]), recoil, 8)

            if i:
                res['amp'][i::2] = interactions['electrons']
            else:
                res['amp'][i::2] = interactions['photons']
                if 'n_excitons' in res.dtype.names:
                    res['n_excitons'][i::2] = interactions['excitons']
            
            
            #Lets just hack some times for now
            
            res['time'][i::2] = interactions['time'] + interactions['t']
            res['endtime'][i::2] = interactions['endtime'] + interactions['t']
        # Remove entries with no quanta
        res = res[res['amp'] > 0]
        return res
        