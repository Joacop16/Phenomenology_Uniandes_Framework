from delphes_reader.particle.abstract_particle import Particle

class ElectronParticle(Particle):
    ''' Particle class object ElectronParticle: 
    
        Attributes: 

        TLV: TLorentzVector that is definning using Pt, Eta, Phi and M.
        Charge: Particle charge.
        Name: Particle Name (electron).
        Type: Particle type (e).
    '''
    def __init__(self, event, j):
        ''' Initialize ElectronParticle extracting attribute values from a delphes file (.root) event.'''
        super().__init__()
        self.TLV.SetPtEtaPhiM(
            event.GetLeaf("Electron.PT").GetValue(j), 
            event.GetLeaf("Electron.Eta").GetValue(j), 
            event.GetLeaf("Electron.Phi").GetValue(j), 
            0.000511 #GeV
        )
        self.Charge = event.GetLeaf("Electron.Charge").GetValue(j)
        self.Name="e"
        self.Type="electron"