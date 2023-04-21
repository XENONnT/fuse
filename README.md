# XeSim
Refactor XENONnT epix and WFsim code using the strax framework. 

Please note: To run the code you need to work on dali. You will need the 'private_nt_aux_files' repo somewhere close by. For all plugins in the detector_physics and pmt_and_daq simulation you need to manualy set the path to the private_nt_aux_files repo.

```
private_files_path = "path/to/private/files"
config = straxen.get_resource(os.path.join(private_files_path, 'sim_files/fax_config_nt_sr0_v4.json') , fmt='json')
```

## Plugin Structure

The full simulation chain in split into multiple plugins. An overview of the simulation structure can be found below.

![Simulation_Refactor_Plugins](https://user-images.githubusercontent.com/27280678/233602026-a91c4ae0-a3ff-4fc9-ac08-2d725bbe8833.jpg)


