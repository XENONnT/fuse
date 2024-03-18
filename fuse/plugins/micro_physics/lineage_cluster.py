import numpy as np
import strax
import logging
import straxen
from numba import njit

import re
import periodictable as pt

export, __all__ = strax.exporter()

from ...plugin import FuseBasePlugin

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.micro_physics.lineage_cluster')

NEST_BETA = (8, 0, 0)
NEST_GAMMA = (7, 0, 0)
NEST_ALPHA = (6, 4, 2)
NEST_NR = (0, 0, 0)
NEST_NONE = (12, 0, 0)


def _print(*args, **kwargs):
    if False:
        print(*args)

@export
class LineageClustering(FuseBasePlugin):
    
    __version__ = "0.0.2"
    
    depends_on = ("geant4_interactions")

    provides = "interaction_lineage"
    
    #output of this plugin will be a index indicating what interaction belongs to what cluster
    #and the lineage type which is the nest model that will later be used to calculate the yields. 
    dtype = [(("Lineage index of the energy deposit", "lineage_index"), np.int32),
             (("Event lineage index", "event_lineage_index"), np.int32),
             (("NEST interaction type", "lineage_type"), np.int32),
             (("Mass number of the interacting particle", "A"), np.int16),
             (("Charge number of the interacting particle", "Z"), np.int16),
             (("Type of the main cluster (alpha beta gamma)", "main_cluster_type"), np.dtype("U10")),
            ]
    dtype = dtype + strax.time_fields

    save_when = strax.SaveWhen.TARGET

    lineages_build = 1

    #Config options
    gamma_distance_threshold = straxen.URLConfig(
        default=0.9, type=(int, float),
        help='Distance threshold to break lineage for gamma rays [cm]. Default taken from NEST code',
    )

    time_threshold = straxen.URLConfig(
        default=10, type=(int, float),
        help='Time threshold to break the lineage [ns]',
    )

    def compute(self, geant4_interactions):
        """
        Args:
            geant4_interactions (np.ndarray): An array of GEANT4 interaction data.

        Returns:
            np.ndarray: An array of cluster IDs with corresponding time and endtime values.
        """

        log.debug(f"Building lineages for {len(geant4_interactions)} interactions")

        if len(geant4_interactions) == 0:
            return np.zeros(0, dtype=self.dtype)

        lineage_ids, lineage_types, lineage_A, lineage_Z, main_cluster_type = self.build_lineages(geant4_interactions)

        #The lineage index is now only unique per event. We need to make it unique for the whole run
        _, unique_lineage_index = np.unique((geant4_interactions["evtid"], lineage_ids),axis = 1 ,return_inverse=True)

        data = np.zeros(len(geant4_interactions), dtype=self.dtype)
        data["lineage_index"] = unique_lineage_index
        data["event_lineage_index"] = lineage_ids
        data["lineage_type"] = lineage_types
        data["A"] = lineage_A
        data["Z"] = lineage_Z
        data["main_cluster_type"] = main_cluster_type

        data["time"] = geant4_interactions["time"]
        data["endtime"] = geant4_interactions["endtime"]


        return data

    def build_lineages(self, geant4_interactions, ):

        event_ids = np.unique(geant4_interactions["evtid"])

        all_lineag_ids = []
        all_lineage_types = []
        all_lineage_As = []
        all_lineage_Zs = []
        all_main_cluster_types = []

        for event_id in event_ids:

            event = geant4_interactions[geant4_interactions["evtid"] == event_id]


            track_id_sort = np.argsort(event[["trackid", "t"]])
            undo_sort_index = np.argsort(track_id_sort)
            event = event[track_id_sort]

            lineage = self.build_lineage_for_event(event)[undo_sort_index]

            _print(f"Event {event_id} has {len(np.unique(lineage['lineage_index']))} lineages ({np.min(lineage['lineage_index'])}, {np.max(lineage['lineage_index'])}) -- lineages build: {self.lineages_build}")

            all_lineag_ids.append(lineage["lineage_index"] + self.lineages_build)
            all_lineage_types.append(lineage["lineage_type"])
            all_lineage_As.append(lineage["lineage_A"])
            all_lineage_Zs.append(lineage["lineage_Z"])
            all_main_cluster_types.append(lineage["main_cluster_type"])

            self.lineages_build = np.max(lineage["lineage_index"])+1

        return np.concatenate(all_lineag_ids), np.concatenate(all_lineage_types), np.concatenate(all_lineage_As), np.concatenate(all_lineage_Zs), np.concatenate(all_main_cluster_types)



    def build_lineage_for_event(self, event):
    
        tmp_dtype = [('lineage_index', np.int32),
                    ('lineage_type', np.int32),
                    ('lineage_A', np.int16),
                    ('lineage_Z', np.int16),
                    ('main_cluster_type', np.dtype("U10")),
                    ]

        _print("-"*50)
        _print(f"Processing event {event[0]['evtid']}")

        tmp_result = np.zeros(len(event), dtype=tmp_dtype)

        main_cluster_type = assign_main_cluster_type_to_event(event)

        # Now iterate all interactions
        running_lineage_index = 0
        for i in range(len(event)):
            
            #Get the particle information
            particle, particle_lineage = get_particle(event, tmp_result, i)
            #Is the particle already in a lineage?
            particle_already_in_lineage = is_particle_in_lineage(particle_lineage)
            #If the particle is not in a lineage, create a new lineage
            if not particle_already_in_lineage:
                #It is the first time we see this particle! Now we need to check if 
                #there is a parent particle.
                parent, parent_lineage = get_parent(event, tmp_result, particle)
                #If there is a parent: 
                if parent is not None:
                    
                    #Evaluate if we have to break the lineage
                    broken_lineage = is_lineage_broken(particle,
                                                       parent,
                                                       parent_lineage,
                                                       self.gamma_distance_threshold,
                                                       self.time_threshold
                                                                )

                    if broken_lineage:
                        #The lineage is broken. We can start a new one!
                        running_lineage_index += 1
                        _print("Broken lineage! Starting new lineage...")
                        tmp_result = start_new_lineage(particle, tmp_result, i, running_lineage_index)
                    
                    else:
                        #The lineage is not broken. We can continue the parent lineage
                        tmp_result = continue_lineage(particle, tmp_result, i, parent_lineage)

                else:
                    #Particle without parent. Start a new lineage
                    running_lineage_index += 1
                    tmp_result = start_new_lineage(particle, tmp_result, i, running_lineage_index)
            
            else:
                #We have seen this particle before. Now evaluate if we have to break the lineage
                last_particle_interaction, last_particle_lineage = get_last_particle_interaction(event, particle, particle_lineage)
                
                #Evaluate if we have to break the lineage
                if last_particle_interaction:
                    broken_lineage = is_lineage_broken(particle,
                                                       last_particle_interaction,
                                                       last_particle_lineage,
                                                       self.gamma_distance_threshold,
                                                       self.time_threshold
                                                        )
                    if broken_lineage:
                        #New lineage!
                        _print("Broken lineage! Starting new lineage...")

                        running_lineage_index += 1
                        tmp_result = start_new_lineage(particle, tmp_result, i, running_lineage_index)

                    else:
                        #The lineage is not broken. We can continue the particle lineage
                        tmp_result = continue_lineage(particle, tmp_result, i, last_particle_lineage)

                else:
                    raise ValueError("There is no last particle interaction but we have seen this particle before.... Makes no sense..")
                    
        tmp_result["main_cluster_type"] = main_cluster_type
        
        return tmp_result


def get_particle(event_interactions, event_lineage, index):
    """
    Returns the particle at the index and the lineage of all interactions of the same particle
    """

    event = event_interactions[index]

    return event, event_lineage[event_interactions["trackid"] == event["trackid"]]

def get_last_particle_interaction(event_interactions, particle, particle_lineage):
    """
    Function to get the last (the previous in time) interaction of the particle (that is in the lineage). 
    """
    
    all_particle_interactions = event_interactions[event_interactions["trackid"] == particle["trackid"]]

    #the last interaction is already in a lineage! Use that: 
    index_of_last_interaction = np.nonzero(particle_lineage)[0][-1]
    return all_particle_interactions[index_of_last_interaction], particle_lineage[index_of_last_interaction]


def get_parent(event_interactions,event_lineage, particle):
    """
    Returns the parent particle and its lineage of the given particle
    """

    index_of_parent_particle = np.where(event_interactions["trackid"] == particle["parentid"])[0]#[0]
    if len(index_of_parent_particle) == 0: #There is no parent particle
        return None, None
    
    parent_interactions = event_interactions[index_of_parent_particle]
    parent_lineages = event_lineage[index_of_parent_particle]

    #Sometimes we can have parents that are after the particle. This makes no sense.
    parent_interactions_time_cut = parent_interactions["t"] <= particle["t"]

    if np.sum(parent_interactions_time_cut) == 0: 
        #there is no parent particle interaction before the particle. Why is this happening? 
        #lets return the parent closest in time.. 
        parent_to_return = np.argmin(abs(parent_interactions["t"] - particle["t"]))
        return parent_interactions[parent_to_return], parent_lineages[parent_to_return]

    #In case there are multiple parent interactions before the particle, we need to take the last one
    possible_parents = parent_interactions[parent_interactions_time_cut]
    possible_parents_lineages = parent_lineages[parent_interactions_time_cut]

    return possible_parents[-1], possible_parents_lineages[-1]

def is_particle_in_lineage(lineage):
    """
    Function to check if a particle is already in a lineage
    """
    
    #All particles in the lineage have not been added to a lineage yet
    if np.all(lineage["lineage_index"] == 0):
        return False
    else: 
        return True

def num_there(s):
    return any(i.isdigit() for i in s)

def classify_lineage(particle_interaction):
    """Function to classify a new lineage based on the particle and its parent information"""

    # NR interactions
    if (particle_interaction["parenttype"] == "neutron") & (num_there(particle_interaction["type"])):
        return NEST_NR

    elif (particle_interaction["parenttype"] == "neutron") & (particle_interaction["type"] == "neutron"):
        return NEST_NR

    #Interactions following a gamma
    elif particle_interaction["parenttype"] == "gamma":
        if particle_interaction["creaproc"] == "compt":
            return NEST_BETA
        elif particle_interaction["creaproc"] == "conv":
            return NEST_BETA
        elif particle_interaction["creaproc"] == "phot":
            return NEST_GAMMA
        else:
            #This case should not happen or? Classify it as nontype
            return NEST_NONE
    
    #Electrons that are not created by a gamma.
    elif particle_interaction["type"] == "e-":
        return NEST_BETA
    
    #The gamma case
    elif particle_interaction["type"] == "gamma":
        if particle_interaction["edproc"] == "compt":
            return NEST_BETA
        elif particle_interaction["edproc"] == "conv":
            return NEST_BETA
        elif particle_interaction["edproc"] == "phot":
            # Not sure about this, but was looking for a way to give beta yields
            # to gammas that are coming directly from a radioactive decay
            if particle_interaction["creaproc"] == "RadioactiveDecayBase":
                return NEST_BETA
            else:
                return NEST_GAMMA
        else:
            #could be rayleigh scattering or something else. Classify it as gamma...
            return NEST_GAMMA
    
    #Primaries and decay products 
    elif (particle_interaction["creaproc"] == "RadioactiveDecayBase") or (particle_interaction["parenttype"] == "none"):

        _print(particle_interaction["type"], particle_interaction["creaproc"], particle_interaction["edproc"], particle_interaction["parenttype"], particle_interaction["trackid"], particle_interaction["ed"])


        #Alpha particles
        if particle_interaction["type"] == "alpha":
            _print("Alpha particle")
            return NEST_ALPHA
            
        #Ions
        elif num_there(particle_interaction["type"]):
            element_number, mass = get_element_and_mass(particle_interaction["type"])
            _print("Ion", element_number, mass)
            return 6, mass, element_number
        
        else:
            #This case should not happen or? Classify it as nontype
            return NEST_NONE
    
    else:
        #No classification possible. Classify it as nontype
        return NEST_NONE
    

@njit()
def is_lineage_broken(particle,
                      parent,
                      parent_lineage,
                      gamma_distance_threshold,
                      time_threshold,
                      ):
    """
    Function to check if the lineage is broken
    """



    #In the nest code: Lineage is always broken if the parent is a ion
    #But if it's an alpha particle, we want to keep the lineage
    brake_for_ion = parent_lineage["lineage_type"] == 6
    # if brake_for_ion:
    #     print("Ion?", parent["type"], particle["type"], parent['creaproc'], parent['edproc'], particle['creaproc'], particle['edproc'])
    brake_for_ion &= (parent["type"] != "alpha")
    brake_for_ion &= (particle['creaproc'] != 'eIoni')
    # brake_for_ion &= (parent["creaproc"] != particle["creaproc"])

    if brake_for_ion:
        # print("Broken for ion")
        return True

    
    #For gamma rays, check the distance between the parent and the particle
    if particle["type"] == "gamma":
        
        #Break the lineage for these transportation gammas
        #Transportations is a special case. They are not real gammas. 
        #They are just used to transport the energy
        #to another volume in the detector (teflon, gas, etc.)
        if parent["edproc"] == "Transportation":
            return True

        particle_position = np.array([particle["x"],particle["y"],particle["z"]])
        parent_position = np.array([parent["x"],parent["y"],parent["z"]])

        distance = np.sqrt(np.sum((parent_position-particle_position)**2, axis=0))

        if distance > gamma_distance_threshold:
            return True

    #I also want to break the lineage if the interaction happens way after the parent interaction
    time_difference = particle["t"] - parent["t"]

    if time_difference > time_threshold:
        return True

    #Does this make sense?
    if (parent["type"] == "neutron"):
        if parent["edproc"].startswith("hadElastic"):
            return True
        elif parent["edproc"].startswith("neutronIne"):
            return True
    
    #Otherwise the lineage is not broken
    return False

def get_element_and_mass(particle_type):
    """
    Function to get the element and the mass number from the particle type
    """

    pattern_match = re.match(r"([a-z]+)([0-9]+)", particle_type, re.I)

    if pattern_match:
        element, mass = pattern_match.groups()
        mass = int(mass)

        element_number = pt.elements.symbol(element).number
    
    else:
        element_number = None
        mass = None
    
    return element_number, mass
    


def assign_main_cluster_type_to_event(event):
    # Initialize columns for processing
    tracks = event['trackid']
    parents = event['parentid']
    types = event['type']
    creaproc = event['creaproc']
    edproc = event['edproc']

    # Initialize mega cluster types with 'None'
    main_cluster_types = np.full(len(tracks), 'None', dtype=object)

    # Directly classify the initial interactions
    is_alpha = (types == 'alpha') | (edproc == 'ionIoni')
    is_beta = (creaproc == 'RadioactiveDecayBase') & (types == 'e-')
    is_gamma = (creaproc == 'RadioactiveDecayBase') & (types == 'gamma')
    is_beta_brem = (creaproc == 'eBrem') & (types == 'gamma')
    
    # Apply initial classifications
    main_cluster_types[is_alpha] = 'alpha'
    main_cluster_types[is_beta] = 'beta'
    main_cluster_types[is_gamma] = 'gamma'

    # Function to propagate a single mega type
    def propagate_mega_type(mega_type):
        # Set of tracks that have been assigned this mega type
        assigned_tracks = set(tracks[main_cluster_types == mega_type])

        previous_assigned_tracks = set()

        while True:

            assigned = set(tracks[main_cluster_types == mega_type])

            # If no new tracks have been assigned, break
            if assigned == previous_assigned_tracks:
                break

            # Update previous assigned tracks
            previous_assigned_tracks = assigned

            # Find all tracks that are children of the assigned tracks
            children_mask = np.in1d(parents, list(assigned))

            # Assign the mega type to the children
            main_cluster_types[children_mask] = mega_type

    # Propagate each mega type
    for mega_type in ['alpha', 'beta', 'gamma']:
        propagate_mega_type(mega_type)
    
    # We need to propagate beta_brem separately, because we overwrite the beta type
    main_cluster_types[is_beta_brem] = 'beta_brem'
    propagate_mega_type('beta_brem')

    return main_cluster_types
    

def start_new_lineage(particle, tmp_result, i, running_lineage_index):

    lineage_class, lineage_A, lineage_Z = classify_lineage(particle)
    tmp_result[i]["lineage_index"] = running_lineage_index
    tmp_result[i]["lineage_type"] = lineage_class
    tmp_result[i]["lineage_A"] = lineage_A
    tmp_result[i]["lineage_Z"] = lineage_Z

    _print("-"*50)
    _print(f"Starting new lineage {running_lineage_index} ({lineage_class}, {lineage_A}, {lineage_Z})")
    _print(f"Type: {particle['type']} Parent type: {particle['parenttype']}")
    _print(f"creaproc: {particle['creaproc']} edproc: {particle['edproc']}")

    return tmp_result

def continue_lineage(particle, tmp_result, i, parent_lineage):
        
        tmp_result[i]["lineage_index"] = parent_lineage["lineage_index"]
        tmp_result[i]["lineage_type"] = parent_lineage["lineage_type"]
        tmp_result[i]["lineage_A"] = parent_lineage["lineage_A"]
        tmp_result[i]["lineage_Z"] = parent_lineage["lineage_Z"]

        return tmp_result