import strax
import straxen
import numpy as np
import logging

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.neutron_veto.nvhitlets')

from ...common import FUSE_PLUGIN_TIMEOUT

#A fuse plugin is a python class that inherits from strax.Plugin
#As naming convention we use CamelCase for the class name
class NeutronVetoHitlets(strax.Plugin):
    
    #Each plugin has a version number
    #If the version number changes, fuse will know that it need to re-simulate the data
    __version__ = "0.0.1"
    
    #You need to tell fuse and strax what the plugin needs as input
    #In this case we need nv_pmthits
    depends_on = ("nv_pmthits")
    
    #You need to tell fuse and strax what the plugin provides as output
    #In this case we provide nv_hitlets
    #You can later use st.make(run_number, "nv_hitlets") to run the simulation
    provides = "nv_hitlets"
    
    #You need to tell fuse and strax what the data looks like
    #Data of the same data_kind can be combined via "horizontal" concatenation and need 
    #to have the same output length. 
    data_kind = 'nv_hitlets'
    
    #You also need to tell strax what columns the data has
    #A column needs a name and a numpy data type. I set everything to float64 here, we can reduce it later
    #I used the columns described here:
    # https://github.com/XENONnT/fuse/blob/f777d9c281d9e046c2f322a30b462a9b7dd8ee00/test_nv/Hitlet_nv_fuse.py#L114
    dtype = [('area', np.float64),
             ('amplitude', np.float64),
             ('time_amplitude', np.float64),
             ('entropy', np.float64),
             ('range_50p_area', np.float64),
             ('range_80p_area', np.float64),
             ('left_area', np.float64),
             ('low_left_area', np.float64),
             ('range_hdr_50p_area', np.float64),
             ('range_hdr_80p_area', np.float64),
             ('left_hdr', np.float64),
             ('low_left_hdr', np.float64),
             ('fwhm', np.float64),
             ('left', np.float64),
             ('fwtm', np.float64),
             ('low_left', np.float64),
            ]
    dtype = dtype + strax.interval_dtype #-> Time, length, dt, channel
    
    #We need to disable automatic rechunking for fuse plugins
    #As fuse is going from "leightweigt" data to "heavy" data,
    #automatic rechunking can lead to problems in later plugins
    rechunk_on_save = False

    #We need to specify when we want to save the data
    save_when = strax.SaveWhen.TARGET

    #strax uses a rather short timeout, lets increase it as 
    #some of the fuse simulation steps can take a while
    input_timeout = FUSE_PLUGIN_TIMEOUT
    
    #We need to tell strax what config options the plugin needs
    #We will use the great URLConfigs that are a part of straxen
    debug = straxen.URLConfig(
        default=False, type=bool,track=False,
        help='Show debug informations',
    )

    deterministic_seed = straxen.URLConfig(
        default=True, type=bool,
        help='Set the random seed from lineage and run_id, or pull the seed from the OS.',
    )
    
    #If you want to prepare something before we start to run the compute method
    #you can put it into the setup method. The setup method is called once while the 
    #compute method is called independently for each chunk
    def setup(self):

        #All plugins can report problmes or debug information via the logging feature
        #You can set the log level via the debug config option. 
        #WARNING messages are always shown whild DEBUG messages are only shown if debug is True
        if self.debug:
            log.setLevel('DEBUG')
            log.debug(f"Running NeutronVetoHitlets version {self.__version__} in debug mode")
        else: 
            log.setLevel('WARNING')

        #Many plugins need to generate random numbers for simulation the corresponding physics process
        #In fuse we want to make sure that the simulation is reproducible.
        #Therefore we have the default setting of deterministic_seed = True
        #In this case the random seed is generated from the run_id and the lineage
        #The lineage includes all plugins and their verions that are connected to the input of the 
        #current plugin as well as all tracked config options and the strax version. 
        #The run_id is a user input. More on the deterministic seed can be found in 
        #a dedicated notebook.
        #Please make sure that you use the random number generator self.rng when you need random numbers
        #later in the plugin. 
        if self.deterministic_seed:
            hash_string = strax.deterministic_hash((self.run_id, self.lineage))
            seed = int(hash_string.encode().hex(), 16)
            self.rng = np.random.default_rng(seed = seed)
            log.debug(f"Generating random numbers from seed {seed}")
        else: 
            self.rng = np.random.default_rng()
            log.debug(f"Generating random numbers with seed pulled from OS")


    #The compute method is the heart of the plugin. It is executed for each chunk of input data and 
    #must produce data in the format specified in the self.dtype variable.
    def compute(self, nv_pmthits):

        #Make sure your plugin can handle empty inputs 
        if len(nv_pmthits) == 0:
            return np.zeros(0, self.dtype)
        
        #All your NV goes here

        
        
        #Build the output array with the correct length and dtype
        # I just return some dummy data here
        result = np.zeros(10, dtype = self.dtype)
        result["time"] = nv_pmthits["time"][0:10]

        return result
