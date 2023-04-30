from Uniandes_Framework import delphes_reader
from Uniandes_Framework.delphes_reader.particle.abstract_particle import Particle
from Uniandes_Framework.delphes_reader.particle.electron_particle import ElectronParticle
from Uniandes_Framework.delphes_reader.particle.jet_particle import JetParticle
from Uniandes_Framework.delphes_reader.particle.met_particle import MetParticle
from Uniandes_Framework.delphes_reader.particle.muon_particle import MuonParticle


jets_NAMES_DIC = {
    'l_jet' : "j",
    'b_jet' : "b",
    'tau_jet' : "#tau",
    'other_jet' : "other"
}

leptons_NAMES_DIC = {
    'electron' : "e",
    'muon' : "#mu"
}

NAMES_DIC = jets_NAMES_DIC.copy()
NAMES_DIC.update(leptons_NAMES_DIC)

#CMS-detector Default Cuts
DEFAULT_CUTS={}
DEFAULT_CUTS["l_jet"]={
    "pt_min_cut":30.,
    "eta_min_cut":-5.,
    "eta_max_cut":+5.
}
DEFAULT_CUTS["b_jet"]={
    "pt_min_cut":30.,
    "eta_min_cut":-2.5,
    "eta_max_cut":+2.5
}
DEFAULT_CUTS["tau_jet"]={
    "pt_min_cut":20.,
    "eta_min_cut":-2.3,
    "eta_max_cut":+2.3
}

DEFAULT_CUTS["other_jet"]={
    "pt_min_cut":0.,
    "eta_min_cut":-5.,
    "eta_max_cut":+5.
}

DEFAULT_CUTS["electron"]={
    "pt_min_cut":25,
    "eta_min_cut":-2.4,
    "eta_max_cut":+2.4
}

DEFAULT_CUTS["muon"]=DEFAULT_CUTS["electron"]

###
def get_met(event):
    ''' Returns a Met Particle.
    Parameters:
        event (TChain): Delphes file (.root) event.
    Return:
        Particle: MetParticles.
    '''
    return MetParticle(event)

def get_unified(part_dic):
    ''' Unifies all particles (contained in part_dic) in a python directory that contains them without any label to separate them.
    Parameters:
        part_dic (Particle python directory): Contains all particles separated by their type as a key of the directory.
    Return:
        Particle (python dictionary): Contains all particles with the key "all".
    '''    
    unified = [particle for particles in part_dic.values() for particle in particles]
    unified.sort(reverse=True, key=Particle.pt)
    return {"all": unified}
    
def get_muons(event):
    ''' Returns a list of muon particles present in a Delphes file event.
    Parameters:
        event (TChain): Delphes file (.root) event.
    Return:
        list: Contains all muon particles.
    '''    
    return [MuonParticle(event, entry) for entry in range(event.Muon.GetEntries())]

def get_electrons(event):
    ''' Returns a list of electron particles present in a Delphes file event.
    Parameters:
        event (TChain): Delphes file (.root) event.
    Return:
        List[ElectronParticle]: List that contains all electrons.
    ''' 
    return [ElectronParticle(event, entry) for entry in range(event.Electron.GetEntries())]

def get_leptons(event):
    ''' Returns a python directory that contains all leptons (muon and electron particles) that are present in an delphes file event.
    Parameters:
        event (TChain): Delphes file (.root) event.
    Return:
        Particle (python directory): Directory that contains all leptons.
    '''
    lepton_dic={
        'electron': get_electrons(event),
        'muon' : get_muons(event)
    }
    
    for key in lepton_dic.keys():
        lepton_dic[key].sort(reverse=True, key=Particle.pt)
    
    return lepton_dic

def get_jets(event):
    '''Returns a dictionary containing all jet particles present in a Delphes file event.
    
    Parameters:
        event (TChain): Delphes file (.root) event.
        
    Returns:
        dict: Contains all jets.
    '''
    jet_dict = {key: [] for key in jets_NAMES_DIC.keys()} 
    
    # classifies the jet particles by their type
    for entry in range(event.Jet.GetEntries()):
        jet = JetParticle(event, entry)
        jet_dict[jet.Type].append(jet)
    
    # sorts the particles in descending order based on their pt attribute
    for key in jet_dict.keys():
        jet_dict[key].sort(reverse=True, key=Particle.pt) # Sort the particles in descending order based on their pt attribute
    
    return jet_dict

def get_good_particles(particles_dic, kin_cuts=DEFAULT_CUTS):
    ''' Returns a dictionary that contains particles within the range of kinematic cuts.
    Parameters:
        particles_dic (dict): Contains all particles separated by their type as a key of the dictionary.
        kin_cuts (dict): Contains dictionaries of kinematic cuts for each particle type. The dictionaries should have the keys "pt_min_cut", "pt_max_cut", "eta_min_cut" and "eta_max_cut".
    Return:
        dict: Contains all particles that are within the range of kinematic cuts.
    '''
    good_particles = {}
    unified_particles = get_unified(particles_dic)["all"]
    
    for particle_type, particles in particles_dic.items():
        if particle_type == "all":
            continue
        name = NAMES_DIC[particle_type]
        good_particles[particle_type] = []
        j = 1
        
        for particle in sorted(particles, reverse=True, key=Particle.pt):
            if not particle.get_good_tag(kin_cuts) == 1:continue
                
            crossed = False
            for check_particle in unified_particles:
                if particle == check_particle: continue
                
                if particle.DeltaR(check_particle) <= 0.3:
                    crossed = True 
                    break
                    
            if crossed: continue
                
            particle.SetName(f"{name}_{{{j}}}")
            good_particles[particle_type].append(particle)
            j += 1
    
    return good_particles

def get_good_leptons(event_particles,kin_cuts=DEFAULT_CUTS):
    ''' Returns a python directory that contains lepton particles that are within the range of kinematic cuts contained in kin_cuts.
    Parameters:
        event_particles (Particle python directory or Tchain event): It could be a dictonary that contains all lepton particles separated by their type as a key of the directory, or a delphes file (.root) event.
        kin_cuts (dict): Contains dictionaries of kinematic cuts for each particle type. Those dictionaries should have the keys "pt_min_cut", "pt_max_cut", "eta_min_cut" and "eta_max_cut".
    Return:
        Particle (dict): Contains all particles that that are within the range of kinematic cuts. This dictionary contains all jet particles separated by their type as a key of the directory.
    '''
    if isinstance(event_particles, dict):
        leptons_dic=event_particles
    else:
        try:
            leptons_dic=get_leptons(event_particles)
        except:
            raise Exception("Error: A dictionary or an event was expected.")
    
    leptons=get_unified(get_good_particles(leptons_dic,kin_cuts))["all"]
    
    for n, lepton in enumerate(leptons):
        j=n+1
        lepton.SetName(f"lep_{{{j}}}")

    return leptons

def get_good_jets(event_particles, kin_cuts=DEFAULT_CUTS):
    '''Returns a dictionary containing jet particles within a range of kinematic cuts.
    
    Parameters:
        event_particles (dict or TChain): A dictionary containing all jet particles separated by their type as a key of the directory, or a Delphes file (.root) event.
        kin_cuts (dict): A dictionary containing dictionaries of kinematic cuts for each particle type. Those dictionaries should have the keys "pt_min_cut", "pt_max_cut", "eta_min_cut", and "eta_max_cut".
        
    Returns:
        dict: Contains all jet particles that are within the range of kinematic cuts, separated by their type as keys in the dictionary.
    '''
    if isinstance(event_particles, dict):
        jets_dict = event_particles
    else:
        jets_dict = get_jets(event_particles)
    
    good_jets_dict = {}
    for key in jets_dict.keys():
        good_jets = []
        for jet in jets_dict[key]:
            if jet.get_good_tag(kin_cuts) == 1:
                good_jets.append(jet)
        good_jets_dict[key] = good_jets
    
    return good_jets_dict