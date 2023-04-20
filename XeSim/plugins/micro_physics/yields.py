import numpy as np
import warnings
import nestpy
import strax
import logging

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('XeSim.micro_physics.yields')
log.setLevel('WARNING')

@strax.takes_config(
    strax.Option('debug', default=False, track=False, infer_type=False,
                 help="Show debug informations"),
)
class NestYields(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ["clustered_interactions", "electric_field_values"]
    provides = "quanta"
    data_kind = "clustered_interactions"
    
    dtype = [('photons', np.float64),
             ('electrons', np.float64),
             ('excitons', np.float64),
            ]
    
    dtype = dtype + strax.time_fields

    #Forbid rechunking
    rechunk_on_save = False
    
    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running NestYields in debug mode")
            log.debug("f'Using nestpy version {nestpy.__version__}'")

        self.quanta_from_NEST = np.vectorize(self._quanta_from_NEST)
    
    def compute(self, clustered_interactions):

        if len(clustered_interactions) == 0:
            return np.zeros(0, dtype=self.dtype)
        
        result = np.zeros(len(clustered_interactions), dtype=self.dtype)
        result["time"] = clustered_interactions["time"]
        result["endtime"] = clustered_interactions["endtime"]

        # Generate quanta:
        if len(clustered_interactions) > 0:
            
            photons, electrons, excitons = self.quanta_from_NEST(clustered_interactions['ed'],
                                                                 clustered_interactions['nestid'],
                                                                 clustered_interactions['e_field'],
                                                                 clustered_interactions['A'],
                                                                 clustered_interactions['Z'],
                                                                 clustered_interactions['create_S2'],
                                                                 density=clustered_interactions['xe_density'])
            
            
            result['photons'] = photons
            result['electrons'] = electrons
            result['excitons'] = excitons
        else:
            result['photons'] = np.empty(0)
            result['electrons'] = np.empty(0)
            result['excitons'] = np.empty(0)
        return result
    
    @staticmethod
    def _quanta_from_NEST(en, model, e_field, A, Z, create_s2, **kwargs):
        """
        Function which uses NEST to yield photons and electrons
        for a given set of parameters.
        Note:
            In case the energy deposit is outside of the range of NEST a -1
            is returned.
        Args:
            en (numpy.array): Energy deposit of the interaction [keV]
            model (numpy.array): Nest Id for qunata generation (integers)
            e_field (numpy.array): Field value in the interaction site [V/cm]
            A (numpy.array): Atomic mass number
            Z (numpy.array): Atomic number
            create_s2 (bool): Specifies if S2 can be produced by interaction,
                in this case electrons are generated.
            kwargs: Additional keyword arguments which can be taken by
                GetYields e.g. density.
        Returns:
            photons (numpy.array): Number of generated photons
            electrons (numpy.array): Number of generated electrons
            excitons (numpy.array): Number of generated excitons
        """
        nc = nestpy.NESTcalc(nestpy.VDetector())
        density = 2.862  # g/cm^3

        # Fix for Kr83m events.
        # Energies have to be very close to 32.1 keV or 9.4 keV
        # See: https://github.com/NESTCollaboration/nest/blob/master/src/NEST.cpp#L567
        # and: https://github.com/NESTCollaboration/nest/blob/master/src/NEST.cpp#L585
        max_allowed_energy_difference = 1  # keV
        if model == 11:
            if abs(en - 32.1) < max_allowed_energy_difference:
                en = 32.1
            if abs(en - 9.4) < max_allowed_energy_difference:
                en = 9.4

        # Some addition taken from
        # https://github.com/NESTCollaboration/nestpy/blob/e82c71f864d7362fee87989ed642cd875845ae3e/src/nestpy/helpers.py#L94-L100
        if model == 0 and en > 2e2:
            log.warning(f"Energy deposition of {en} keV beyond NEST validity for NR model of 200 keV - Remove Interaction")
            return -1, -1, -1
        if model == 7 and en > 3e3:
            log.warning(f"Energy deposition of {en} keV beyond NEST validity for gamma model of 3 MeV - Remove Interaction")
            return -1, -1, -1
        if model == 8 and en > 3e3:
            log.warning(f"Energy deposition of {en} keV beyond NEST validity for beta model of 3 MeV - Remove Interaction")
            return -1, -1, -1

        y = nc.GetYields(interaction=nestpy.INTERACTION_TYPE(model),
                         energy=en,
                         drift_field=e_field,
                         A=A,
                         Z=Z,
                         **kwargs
                         )

        event_quanta = nc.GetQuanta(y)  # Density argument is not use in function...

        photons = event_quanta.photons
        excitons = event_quanta.excitons
        electrons = 0
        if create_s2:
            electrons = event_quanta.electrons

        return photons, electrons, excitons
    
    

@strax.takes_config(
    strax.Option('debug', default=False, track=False, infer_type=False,
                 help="Show debug informations"),
)
class BBFYields(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ["clustered_interactions", "electric_field_values"]
    provides = "quanta"
    
    dtype = [('photons', np.float64),
             ('electrons', np.float64),
             ('excitons', np.float64),
            ]
    
    dtype = dtype + strax.time_fields

    def setup(self):
        self.bbfyields = BBF_quanta_generator()

        if self.debug:
            log.setLevel("DEBUG")
            log.debug("Running BBFYields in debug mode")

    def compute(self, geant4_interactions):
        
        result = np.zeros(len(geant4_interactions), dtype=self.dtype)
        result["time"] = geant4_interactions["time"]
        result["endtime"] = geant4_interactions["endtime"]

        # Generate quanta:
        if len(geant4_interactions) > 0:

            photons, electrons, excitons = self.bbfyields.get_quanta_vectorized(
                                energy=geant4_interactions['ed'],
                                interaction=geant4_interactions['nestid'],
                                field=geant4_interactions['e_field']
                                )

            
            
            result['photons'] = photons
            result['electrons'] = electrons
            result['excitons'] = excitons
        else:
            result['photons'] = np.empty(0)
            result['electrons'] = np.empty(0)
            result['excitons'] = np.empty(0)
        return result
    

class BBF_quanta_generator:
    def __init__(self):
        self.er_par_dict = {
            'W': 0.013509665661431896,
            'Nex/Ni': 0.08237994367314523,
            'py0': 0.12644250072199228,
            'py1': 43.12392476032283,
            'py2': -0.30564651066249543,
            'py3': 0.937555814189728,
            'py4': 0.5864910020458629,
            'rf0': 0.029414125811261564,
            'rf1': 0.2571929264699089,
            'fano' : 0.059
        }
        self.nr_par_dict = {
            "W": 0.01374615297291325, 
            "alpha": 0.9376149722771664, 
            "zeta": 0.0472,
            "beta": 311.86846286764376, 
            "gamma": 0.015772527423653895, 
            "delta": 0.0620,
            "kappa": 0.13762801393921467, 
            "eta": 6.387273512457444, 
            "lambda": 1.4102590741165675,
            "fano" : 0.059
        }
        self.ERs = [7,8,11]
        self.NRs = [0,1]
        self.unknown = [12]
        self.get_quanta_vectorized = np.vectorize(self.get_quanta, excluded="self")
        
    def update_ER_params(self, new_params):
        self.er_par_dict.update(new_params)
    def update_NR_params(self, new_params):
        self.nr_par_dict.update(new_params)

    def get_quanta(self, interaction, energy, field):
        if int(interaction) in self.ERs: 
            return self.get_ER_quanta(energy, field, self.er_par_dict)
        elif int(interaction) in self.NRs:
            return self.get_NR_quanta(energy, field, self.nr_par_dict)
        elif int(interaction) in self.unknown:
            return 0,0,0
        else:
            raise RuntimeError("Unknown nest ID: {:d}, {:s}".format(
                            int(interaction), 
                            str(nestpy.INTERACTION_TYPE(int(interaction)))))
        
    ####
    def ER_recomb(self, energy, field, par_dict):
        W = par_dict['W']
        ExIonRatio = par_dict['Nex/Ni']

        Nq = energy / W
        Ni = Nq / (1. + ExIonRatio)
        Nex = Nq - Ni

        TI = par_dict['py0'] * np.exp(-energy/par_dict['py1']) * field**par_dict['py2']
        Recomb = 1. - np.log(1. + TI*Ni/4.) / (TI*Ni/4.)
        FD = 1. / (1. + np.exp(-(energy-par_dict['py3'])/par_dict['py4']))

        return Recomb * FD
    def ER_drecomb(self, energy, par_dict):
        return par_dict['rf0'] * (1. - np.exp(-energy/par_dict['py1']))
    
    def NR_quenching(self, energy, par_dict):
        alpha = par_dict['alpha']
        beta = par_dict['beta']
        gamma = par_dict['gamma']
        delta = par_dict['delta']
        kappa = par_dict['kappa']
        eta = par_dict['eta']
        lam = par_dict['lambda']
        zeta = par_dict['zeta']

        e = 11.5 * energy * 54.**(-7./3.)
        g = 3. * e**0.15 + 0.7 * e**0.6 + e

        return kappa*g / (1. + kappa*g)
    def NR_ExIonRatio(self, energy, field, par_dict):
        alpha = par_dict['alpha']
        beta = par_dict['beta']
        gamma = par_dict['gamma']
        delta = par_dict['delta']
        kappa = par_dict['kappa']
        eta = par_dict['eta']
        lam = par_dict['lambda']
        zeta = par_dict['zeta']

        e = 11.5 * energy * 54.**(-7./3.)

        return alpha * field**(-zeta) * (1. - np.exp(-beta*e))
    def NR_Penning_quenching(self, energy, par_dict):
        alpha = par_dict['alpha']
        beta = par_dict['beta']
        gamma = par_dict['gamma']
        delta = par_dict['delta']
        kappa = par_dict['kappa']
        eta = par_dict['eta']
        lam = par_dict['lambda']
        zeta = par_dict['zeta']

        e = 11.5 * energy * 54.**(-7./3.)
        g = 3. * e**0.15 + 0.7 * e**0.6 + e

        return 1. / (1. + eta * e**lam)
    def NR_recomb(self, energy, field, par_dict):
        alpha = par_dict['alpha']
        beta = par_dict['beta']
        gamma = par_dict['gamma']
        delta = par_dict['delta']
        kappa = par_dict['kappa']
        eta = par_dict['eta']
        lam = par_dict['lambda']
        zeta = par_dict['zeta']

        e = 11.5 * energy * 54.**(-7./3.)
        g = 3. * e**0.15 + 0.7 * e**0.6 + e

        HeatQuenching = self.NR_quenching(energy, par_dict)
        PenningQuenching = self.NR_Penning_quenching(energy, par_dict)

        ExIonRatio = self.NR_ExIonRatio(energy, field, par_dict)

        xi = gamma * field**(-delta)
        Nq = energy * HeatQuenching / par_dict['W']
        Ni = Nq / (1.+ ExIonRatio)

        return 1. - np.log(1. + Ni*xi) / (Ni*xi)
    ###
    def get_ER_quanta(self, energy, field, par_dict):
        Nq_mean = energy / par_dict['W']
        Nq = np.clip(np.round(np.random.normal(Nq_mean, np.sqrt(Nq_mean * par_dict['fano']))), 0, np.inf).astype(np.int64)

        Ni = np.random.binomial(Nq, 1./(1.+par_dict['Nex/Ni']))

        recomb = self.ER_recomb(energy,field, par_dict)
        drecomb = self.ER_drecomb(energy, par_dict)
        true_recomb = np.clip(np.random.normal(recomb, drecomb), 0., 1.)

        Ne = np.random.binomial(Ni, 1.-true_recomb)
        Nph = Nq - Ne
        Nex = Nq - Ni
        return Nph, Ne, Nex
    def get_NR_quanta(self, energy, field, par_dict):
        Nq_mean = energy / par_dict['W']
        Nq = np.round(np.random.normal(Nq_mean, np.sqrt(Nq_mean * par_dict['fano']))).astype(np.int64)

        quenching = self.NR_quenching(energy, par_dict)
        Nq = np.random.binomial(Nq, quenching)

        ExIonRatio = self.NR_ExIonRatio(energy, field,par_dict)
        Ni = np.random.binomial(Nq, ExIonRatio/(1.+ExIonRatio))

        penning_quenching = self.NR_Penning_quenching(energy, par_dict)
        Nex = np.random.binomial(Nq - Ni, penning_quenching)

        recomb = self.NR_recomb(energy, field, par_dict)
        if recomb < 0 or recomb > 1:
            return None, None

        Ne = np.random.binomial(Ni, 1.-recomb)
        Nph = Ni + Nex - Ne
        return Nph, Ne, Nex
    