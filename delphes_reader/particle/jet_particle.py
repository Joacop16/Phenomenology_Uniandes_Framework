from Uniandes_Framework.delphes_reader.particle.abstract_particle import Particle
import random

JET_TYPES={
        "BTag0_TauTag0":"l_jet",
        "BTag1_TauTag0":"b_jet",
        "BTag0_TauTag1":"tau_jet"
}

    
class JetParticle(Particle):
    ''' Particle class object JetParticle: 
    
        Attributes: 

        TLV: TLorentzVector that is defining using Pt, Eta, Phi and Mass.
        Charge: Particle charge.
        Name: Particle Name (examples: l_jet_1, l_jet_2, etc.).
        Type: Particle type (examples: l_jet, b_jet, tau_jet, other_jet).
        BTag: Particle BTag, it could be 0 or 1.
        TauTag: Particle TauTag, it could be 0 or 1.
        CTag: Particle CTag, it could be 0 or 1.
        Flavor: Particle flavor, it could be 4 for a charm quark or 5 for a bottom quark.
    
    Methods:
        _jet_type: Estimates the jet type based on the BTag and TauTag attributes.
        c_tagging: Determines the CTag attribute based on the Flavor attribute and a random number.
    
    '''
    
    def __init__(self, event, j):
        ''' Initialize JetParticle extracting attribute values from a delphes file (.root) event.
        
        Parameters:
            event (pyroot.TTree): Delphes event containing particle information.
            j (int): Jet index in the Delphes event.
            
        '''
        super().__init__()
        self.TLV.SetPtEtaPhiM(
            event.GetLeaf("Jet.PT").GetValue(j), 
            event.GetLeaf("Jet.Eta").GetValue(j), 
            event.GetLeaf("Jet.Phi").GetValue(j), 
            event.GetLeaf("Jet.Mass").GetValue(j)
        )
        self.Flavor = int(event.GetLeaf("Jet.Flavor").GetValue(j))
        self.CTag   = int(self.c_tagging())
        self.BTag   = int(event.GetLeaf("Jet.BTag").GetValue(j))
        self.TauTag = int(event.GetLeaf("Jet.TauTag").GetValue(j))
        self.Charge = event.GetLeaf("Jet.Charge").GetValue(j)
        self.Type=self._jet_type()
        self.Name=f'{self.Type}_{{{j}}}'
        
    def _jet_type(self):
        ''' Estimates the jet type based on the BTag and TauTag attributes.
        
        Returns:
            str: particle type.
        '''
        return JET_TYPES.get(f'BTag{self.BTag}_TauTag{self.TauTag}',"other_jet")
    
    def c_tagging(self):
        ''' Determines the CTag attribute based on the Flavor attribute and a random number.

        For charm jets, the method returns 1 with a 70% probability, and for all other jets, it returns 1 with a 1% probability.

        Returns:
            int: CTag, which could be 0 or 1.
        '''
        random_number = random.random()

        if(self.Flavor == 4 and random_number < 0.7): return 1
        elif(self.Flavor != 4 and random_number < 0.01): return 1
        else: return 0
