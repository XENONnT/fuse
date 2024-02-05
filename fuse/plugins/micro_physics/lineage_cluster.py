import numpy as np
import strax
import logging
import straxen

import re
import periodictable as pt

export, __all__ = strax.exporter()

from ...plugin import FuseBasePlugin

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.micro_physics.lineage_cluster')

@export
class LineageClustering(FuseBasePlugin):
    
    __version__ = "0.0.1"
    
    depends_on = ("geant4_interactions")

    provides = "interaction_lineage"
    
    #output of this plugin will be a index indicating what interaction belongs to what cluster
    #and the lineage type which is the nest model that will later be used to calculate the yields. 
    dtype = [('lineage_index', np.int32),
             ('lineage_type', np.int32),
             ('A', np.int16), #Atomic mass
             ('Z', np.int16), #Atomic number

            ]
    dtype = dtype + strax.time_fields

    save_when = strax.SaveWhen.TARGET

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
        if len(geant4_interactions) == 0:
            return np.zeros(0, dtype=self.dtype)

        lineage_ids, lineage_types, lineage_A, lineage_Z = self.build_lineages(geant4_interactions)

        #The lineage index is now only unique per event. We need to make it unique for the whole run
        _, unique_lineage_index = np.unique((geant4_interactions["evtid"], lineage_ids),axis = 1 ,return_inverse=True)

        data = np.zeros(len(geant4_interactions), dtype=self.dtype)
        data["lineage_index"] = unique_lineage_index
        data["lineage_type"] = lineage_types
        data["A"] = lineage_A
        data["Z"] = lineage_Z

        data["time"] = geant4_interactions["time"]
        data["endtime"] = geant4_interactions["endtime"]

        return data

    def build_lineages(self, geant4_interactions):

        event_ids = np.unique(geant4_interactions["evtid"])

        all_lineag_ids = []
        all_lineage_types = []
        all_lineage_As = []
        all_lineage_Zs = []

        for event_id in event_ids:
            event = geant4_interactions[geant4_interactions["evtid"] == event_id]

            lineage = self.build_lineage_for_event(event)
            #lineage_index, lineage_type, lineage_A, lineage_Z = self.build_lineage_for_event(event)

            all_lineag_ids.append(lineage["lineage_index"])
            all_lineage_types.append(lineage["lineage_type"])
            all_lineage_As.append(lineage["lineage_A"])
            all_lineage_Zs.append(lineage["lineage_Z"])
        
        return np.concatenate(all_lineag_ids), np.concatenate(all_lineage_types), np.concatenate(all_lineage_As), np.concatenate(all_lineage_Zs)

    def build_lineage_for_event(self, event):

        tmp_dtype = [('lineage_index', np.int32),
                     ('lineage_type', np.int32),
                     ('lineage_A', np.int16),
                     ('lineage_Z', np.int16),
                    ]
        tmp_result = np.zeros(len(event), dtype=tmp_dtype)

        running_lineage_index = 0

        #Iterate over all interactions in the event
        for i in range(len(event)):

            #Get the particle and its lineage
            particle, particle_lineage = get_particle(event, tmp_result, i)

            #Get the parent particle (if there is one) and its lineage  
            parent, parent_lineage = get_parent(event, tmp_result, particle)

            #Replace the parent in case a particle has multiple interactions in the same event
            #if (parent is not None):
            #    if len(parent_lineage) > 1:
            #        parent, parent_lineage = get_previous_parent_interaction(parent, parent_lineage, particle)
            #    else:
            #        parent = parent[0]

            if parent is None: #If there is no parent particle

                #Check if the particle is already in a lineage
                particle_already_in_lineage = is_particle_in_lineage(particle_lineage)

                if particle_already_in_lineage:
                    
                    #Check if the lineage will be broken -> a gamma scattering at multiple places
                    last_particle_interaction, last_particle_lineage = get_last_particle_interaction(event, particle, particle_lineage)

                    if last_particle_interaction:
                        #There is a last interaction!
                        #Check if the lineage is broken
                        broken_lineage = is_lineage_broken(particle,
                                                           last_particle_interaction,
                                                           last_particle_lineage,
                                                           gamma_distance_threshold = self.gamma_distance_threshold,
                                                           time_threshold = self.time_threshold
                                                           )
                        if broken_lineage:
                            #New lineage
                            running_lineage_index += 1
                            lineage_class, lineage_A, lineage_Z = lineage_classification(particle, broken_lineage = True)

                            tmp_result[i]["lineage_index"] = running_lineage_index
                            tmp_result[i]["lineage_type"] = lineage_class
                            tmp_result[i]["lineage_A"] = lineage_A
                            tmp_result[i]["lineage_Z"] = lineage_Z
                        
                        else:
                            #Add the particle to the lineage
                            tmp_result[i]["lineage_index"] = last_particle_lineage["lineage_index"]#[0]
                            tmp_result[i]["lineage_type"] = last_particle_lineage["lineage_type"]#[0]
                            tmp_result[i]["lineage_A"] = last_particle_lineage["lineage_A"]#[0]
                            tmp_result[i]["lineage_Z"] = last_particle_lineage["lineage_Z"]#[0]
                        continue
                        

                    #If the particle has no last interaction, or the lineage is not broken, we can add the particle to the lineage
                    tmp_result[i]["lineage_index"] = particle_lineage["lineage_index"][0]
                    tmp_result[i]["lineage_type"] = particle_lineage["lineage_type"][0]
                    tmp_result[i]["lineage_A"] = particle_lineage["lineage_A"][0]
                    tmp_result[i]["lineage_Z"] = particle_lineage["lineage_Z"][0]
                
                else:
                    #New lineage
                    running_lineage_index += 1
                    lineage_class, lineage_A, lineage_Z = lineage_classification(particle, broken_lineage = True)

                    tmp_result[i]["lineage_index"] = running_lineage_index
                    tmp_result[i]["lineage_type"] = lineage_class
                    tmp_result[i]["lineage_A"] = lineage_A
                    tmp_result[i]["lineage_Z"] = lineage_Z
                continue
            
            #If there is a parent particle, we need to check if the particle is already in a lineage
            particle_already_in_lineage = is_particle_in_lineage(particle_lineage)
            if particle_already_in_lineage:
                #Check if the lineage will be broken -> a gamma scattering at multiple places
                last_particle_interaction, last_particle_lineage = get_last_particle_interaction(event, particle, particle_lineage)

                if last_particle_interaction:
                    #There is a last interaction!
                    #Check if the lineage is broken
                    broken_lineage = is_lineage_broken(particle,
                                                        last_particle_interaction,
                                                        last_particle_lineage,
                                                        gamma_distance_threshold = self.gamma_distance_threshold,
                                                        time_threshold = self.time_threshold
                                                        )
                    if broken_lineage:
                        #New lineage
                        running_lineage_index += 1
                        lineage_class, lineage_A, lineage_Z = lineage_classification(particle, broken_lineage = True)

                        tmp_result[i]["lineage_index"] = running_lineage_index
                        tmp_result[i]["lineage_type"] = lineage_class
                        tmp_result[i]["lineage_A"] = lineage_A
                        tmp_result[i]["lineage_Z"] = lineage_Z
                        
                    else:
                        #Add the particle to the lineage
                        tmp_result[i]["lineage_index"] = last_particle_lineage["lineage_index"]#[0]
                        tmp_result[i]["lineage_type"] = last_particle_lineage["lineage_type"]#[0]
                        tmp_result[i]["lineage_A"] = last_particle_lineage["lineage_A"]#[0]
                        tmp_result[i]["lineage_Z"] = last_particle_lineage["lineage_Z"]#[0]
                    continue


            #There is no parent particle and the particle is not in a lineage yet: 
            #Check if the lineage is broken
            broken_lineage = is_lineage_broken(particle,
                                               parent,
                                               parent_lineage,
                                               gamma_distance_threshold = self.gamma_distance_threshold,
                                               time_threshold = self.time_threshold
                                               )

            if broken_lineage:

                #If the lineage was broken due to the ion case and the particle was already put into a lineage
                #if (parent_lineage["lineage_type"] == 6) & is_particle_in_lineage(particle_lineage):
                    #Add the particle to the lineage
                #    tmp_result[i]["lineage_index"] = particle_lineage["lineage_index"][0]
                #    tmp_result[i]["lineage_type"] = particle_lineage["lineage_type"][0]
                #    tmp_result[i]["lineage_A"] = particle_lineage["lineage_A"][0]
                #    tmp_result[i]["lineage_Z"] = particle_lineage["lineage_Z"][0]

                #else:
                    #New lineage
                running_lineage_index += 1
                lineage_class, lineage_A, lineage_Z = lineage_classification(particle, broken_lineage)
                    
                tmp_result[i]["lineage_type"] = lineage_class
                tmp_result[i]["lineage_index"] = running_lineage_index
                tmp_result[i]["lineage_A"] = lineage_A
                tmp_result[i]["lineage_Z"] = lineage_Z


            else:# If the lineage is not broken, add the particle to the parents lineage
                tmp_result[i]["lineage_type"] = parent_lineage["lineage_type"]
                tmp_result[i]["lineage_index"] = parent_lineage["lineage_index"]
                tmp_result[i]["lineage_A"] = parent_lineage["lineage_A"]
                tmp_result[i]["lineage_Z"] = parent_lineage["lineage_Z"]

        return tmp_result#tmp_result["lineage_index"], tmp_result["lineage_type"]


def get_particle(event_interactions, event_lineage, index):
    """
    Returns the particle at the index and the lineage of all interactions of the same particle
    """

    event = event_interactions[index]

    return event, event_lineage[event_interactions["trackid"] == event["trackid"]]


def get_last_particle_interaction(event_interactions, particle, particle_lineage):
    """
    Function to get the last interaction of a particle
    """
    
    all_particle_interactions = event_interactions[event_interactions["trackid"] == particle["trackid"]]
    particle_mask = all_particle_interactions == particle
    last_interaction_index = np.where(particle_mask)[0][0]-1

    if last_interaction_index < 0:
        return None, None
    else:
        return all_particle_interactions[last_interaction_index], particle_lineage[last_interaction_index]



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
        #lets return the first parent interaction..
        #parent_to_return = np.argmin(parent_interactions["t"])
        #return parent_interactions[parent_to_return], parent_lineages[parent_to_return]

        return None, None

    #In case there are multiple parent interactions before the particle, we need to take the last one
    possible_parents = parent_interactions[parent_interactions_time_cut]
    possible_parents_lineages = parent_lineages[parent_interactions_time_cut]

    return possible_parents[-1], possible_parents_lineages[-1]
    
    #return parent_interactions[parent_interactions_time_cut], parent_lineages[parent_interactions_time_cut]

def get_previous_parent_interaction(parent_interactions, parent_lineages, particle):
    """
    Returns the parent interaction before the current particle

    Merge this function with get_parent... 

    """

    possible_parents_mask = parent_interactions["t"] <= particle["t"]

    possible_parents = parent_interactions[possible_parents_mask]
    possible_parents_lineages = parent_lineages[possible_parents_mask]

    assert len(possible_parents) > 0

    return possible_parents[-1], possible_parents_lineages[-1]

def is_particle_in_lineage(lineage):
    """
    Function to check if a particle is already in a lineage
    """

    lineage_index_values = np.unique(lineage["lineage_index"])
    
    #All particles in the lineage have not been added to a lineage yet
    if np.all(lineage_index_values == 0):
        return False

    else: 
        return True


def num_there(s):
    return any(i.isdigit() for i in s)

def lineage_classification(particle_interaction, broken_lineage = False):
    """
    Function to classifiy a new lineage based on the particle and its parent information

    Returns: 
        lineage_class: The lineage class according to the nest model
        A: The atomic mass of the particle
        Z: The atomic number of the particle
    """
    #Missing the Kr83m case

    if (particle_interaction["parenttype"] == "neutron") & (num_there(particle_interaction["type"])):
        
        return 0, 0 , 0

    elif (particle_interaction["parenttype"] == "neutron") & (particle_interaction["type"] == "neutron"):
        
        return 0, 0 , 0

    elif particle_interaction["parenttype"] == "gamma":
        if particle_interaction["creaproc"] == "compt":
            return 8, 0 , 0
        elif particle_interaction["creaproc"] == "conv":
            return 8, 0 , 0
        elif particle_interaction["creaproc"] == "phot":
            return 7, 0 , 0
        else:
            return 12, 0 , 0
    
    elif particle_interaction["type"] == "e-": #more cases to consider for beta decays??
        return 8, 0 , 0
    
    #The ion case #Only here A and Z are needed 
    elif (particle_interaction["creaproc"] == "Radioactiv") or (particle_interaction["parenttype"] == "none"):
        if num_there(particle_interaction["type"]):
            
            element_number, mass = get_element_and_mass(particle_interaction["type"])

            return 6, mass, element_number   #Get the correct values here!

        elif particle_interaction["type"] == "alpha":
            return 6, 4, 2

        #Maybe add the information that the lineage is broken to the function. 
        elif particle_interaction["type"] == "gamma":
            if particle_interaction["edproc"] == "compt":
                return 8, 0 , 0
            elif particle_interaction["edproc"] == "conv":
                return 8, 0 , 0
            elif particle_interaction["edproc"] == "phot":
                return 7, 0 , 0
            else:
                return 12, 0 , 0

        else:
            return 12, 0 , 0
        
    #A lineage can only be broken if we have a ion (see last case or if we have a gamma. This case)
    elif broken_lineage:

        #Maybe add the information that the lineage is broken to the function. 
        if particle_interaction["type"] == "gamma":
            if particle_interaction["edproc"] == "compt":
                return 8, 0 , 0
            elif particle_interaction["edproc"] == "conv":
                return 8, 0 , 0
            elif particle_interaction["edproc"] == "phot":
                return 7, 0 , 0

            else:
                return 12, 0 , 0
        
        else:
            return 12, 0 , 0

    else: 
        return 12, 0 , 0
    


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
    if parent_lineage["lineage_type"] == 6:
        return True
    
    #For gamma rays, check the distance between the parent and the particle
    if particle["type"] == "gamma":
        
        #Break the lineage for these transportation gammas
        if parent["edproc"] == "Transporta":
            return True

        particle_position = np.array([particle["x"],particle["y"],particle["z"]])
        parent_position = np.array([parent["x"],parent["y"],parent["z"]])

        distance = np.sqrt(np.sum((parent_position-particle_position)**2, axis=0))

        if distance > gamma_distance_threshold:
            return True

    # I also want to break the lineage if the interaction happens way after the parent interaction
    time_difference = particle["t"] - parent["t"]

    if time_difference > time_threshold:
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
        print("No Match - Should not happen!")
        element_number = None
        mass = None
    
    return element_number, mass
    
    