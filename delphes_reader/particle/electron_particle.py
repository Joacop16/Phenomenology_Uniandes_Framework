from Uniandes_Framework.delphes_reader.particle.abstract_particle import Particle
       
class ElectronParticle(Particle):
    ''' A class representing an electron particle.

    Attributes:
        TLV (TLorentzVector): A TLorentzVector object that defines the electron's momentum and energy.
        Charge (float): The electric charge of the electron.
        Name (str): The name of the particle ("e" for electron).
        Type (str): The type of the particle ("electron" for an electron).

    Methods:
        SetCharge(charge):
            Sets the electric charge of the electron.

    Example:
        To create an ElectronParticle object using data from a delphes file event:

        >>> event = ROOT.TChain("Delphes")
        >>> event.Add("events.root")
        >>> electron = ElectronParticle(event, 0)
    '''
    
    def __init__(self, event, j):
        ''' Initializes an ElectronParticle object using data extracted from a delphes file event.

        Args:
            event (ROOT.TTree): A ROOT.TTree object containing the event data.
            j (int): An integer representing the index of the electron particle in the event.

        '''
        super().__init__()
        self.TLV.SetPtEtaPhiM(
            event.GetLeaf("Electron.PT").GetValue(j), 
            event.GetLeaf("Electron.Eta").GetValue(j), 
            event.GetLeaf("Electron.Phi").GetValue(j), 
            0.000511 # GeV
        )
        self.Charge = event.GetLeaf("Electron.Charge").GetValue(j)
        self.Name="e"
        self.Type="electron"
