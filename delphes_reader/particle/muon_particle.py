from Uniandes_Framework.delphes_reader.particle.abstract_particle import Particle

class MuonParticle(Particle):
    ''' Particle class object MuonParticle: 
    
        Attributes: 

        TLV: TLorentzVector that is definning using Pt, Eta, Phi and M.
        Charge: Particle charge.
        Name: Particle Name (#mu).
        Type: Particle type (muon).
    '''
    def __init__(self, event, j):
        ''' Initialize MuonParticle extracting attribute values from a delphes file (.root) event.'''
        super().__init__()
        self.TLV.SetPtEtaPhiM(
            event.GetLeaf("Muon.PT").GetValue(j), 
            event.GetLeaf("Muon.Eta").GetValue(j), 
            event.GetLeaf("Muon.Phi").GetValue(j), 
            0.1056583755 #GeV
        )
        self.Charge = event.GetLeaf("Muon.Charge").GetValue(j)
        self.Name="#mu"
        self.Type="muon"