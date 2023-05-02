from abc import ABC
from ROOT import (
    TLorentzVector,
    TVector3,
    TVector2,
    TMath
)

class Particle(ABC):
    '''
    Abstract base class for particles in high-energy physics experiments.

    Attributes:
    -----------
    TLV : TLorentzVector
        A TLorentzVector representing the four-momentum of the particle, defined by its
        transverse momentum (Pt), pseudorapidity (Eta), azimuthal angle (Phi), and mass (M).
    Charge : float
        The electric charge of the particle.
    Name : str
        The name of the particle, e.g. "#mu", "e", "MET", etc.
    Type : str
        The type of the particle, e.g. "muon", "electron", "MET", etc.

    Methods:
    --------
    GetCharge() -> float:
        Returns the electric charge of the Particle object.

    GetTLV() -> TLorentzVector:
        Returns the 4-momentum of the Particle object as a TLorentzVector.

    GetName() -> str:
        Returns the name of the Particle object.

    SetName(new_name: str) -> None:
        Sets the name of the Particle object to the new name provided as an argument.

    set_good_tag(value: int) -> None:
        Sets the GoodTag label of the particle.
        
    get_good_tag(cuts: dict) -> int:
        Defines and returns a label (GoodTag) that indicates if the particle is within the range of kinematic cuts (Pt_min, Pt_max, Eta_min, Eta_max).
        GoodTag could have two values:
            1: Particle is within the range of kinematic cuts.
            0: Particle is not within the range of kinematic cuts.
        Parameters:
            cuts (dict): A dictionary containing the values of kinematic cuts. It should have the keys "pt_min_cut", "pt_max_cut" (optional), "eta_min_cut" and "eta_max_cut".
        Returns:
        int: GoodTag.
    SetCharge ()
    
    DeltaR ()
    
    DeltaEta ()
    
    DeltaPhi ()
    
    sDeltaPT ()
    
    vDeltaPT ()
    
    vDeltaP ()
    
    Properties:
    --------
    p : float
        Returns the full momentum (P) of the particle, which is the magnitude of the three-momentum vector.
    
    pl : float
        Returns the longitudinal momentum (P_L) of the particle, which is the component of the momentum parallel to the beam direction.
    
    eta : float
        Returns the pseudorapidity (Eta) of the particle, which is related to the polar angle of the momentum vector in the detector frame.
    
    phi : float
        Returns the azimuthal angle (Phi) of the particle, which is the angle between the transverse momentum vector and a reference axis in the transverse plane.
    
    m : float
        Returns the reconstructed mass (M) of the particle, which is calculated from the four-momentum vector.
    
    energy : float
        Returns the reconstructed energy (E) of the particle, which is the time component of the four-momentum vector.

    Notes:
    ------
    This is an abstract base class that cannot be instantiated on its own. It serves as a 
    blueprint for subclasses that inherit its attributes and methods. Subclasses should 
    implement their own methods according to their specific use case.
    '''
  
    def __init__(self, **kwargs):
        '''Initializes a new Particle object.

        Keyword Args:
            charge (float): The particle's electric charge (default: 0.0).
            name (str): The particle's name (default: "").
            particle_type (str): The particle's type (default: "").

        Raises:
            TypeError: If any of the arguments is of the wrong type.
        '''
        self.TLV = TLorentzVector()
        self.Charge = kwargs.get("charge", 0.0)
        self.Name = kwargs.get("name", "")
        self.Type = kwargs.get("particle_type", "")

        if not isinstance(self.TLV, TLorentzVector):
            raise TypeError("TLV must be a TLorentzVector object")
        if not isinstance(self.Charge, float):
            raise TypeError("Charge must be a float")
        if not isinstance(self.Name, str):
            raise TypeError("Name must be a string")
        if not isinstance(self.Type, str):
            raise TypeError("Type must be a string")
            
    def GetCharge(self):
        ''' Returns charge (attribute) of particle.
        
        Returns:
            float: Charge.'''
        charge=self.Charge
        return charge
    
    def GetTLV (self):
        ''' Returns TLV (attribute) of particle (this TLV is definning with Pt, Eta, Phi and M).
        
        Returns:
            TLorentzVector: TLV.
        '''
        tlv=self.TLV
        return tlv

    def GetName(self):
        ''' Returns name (attribute) of particle
        
        Returns:
            str: Name.
        '''
        name=self.Name
        return name
    
    def SetName(self, new_name: str) -> None:
        '''Sets the name of the Particle object to the new name provided as an argument.

        Args:
            new_name (str): The new name to assign to the particle.

        Returns:
            None
        '''
        self.Name = new_name

    @property
    def pt(self) -> float:
        '''Returns the transverse momentum (Pt) of the Particle object.

        Returns:
            float: The transverse momentum (Pt) of the particle.
        '''
        TLV = self.TLV
        return TLV.Pt()

    @property
    def p(self) -> float:
        '''Returns the full momentum (P) of the Particle object.

        Returns:
            float: The full momentum (P) of the particle.
        '''
        TLV = self.TLV
        return TLV.P()

    @property
    def pl(self) -> float:
        '''Returns the longitudinal momentum of the Particle object.

        The longitudinal momentum is estimated using the full momentum (P) and transverse momentum (Pt).

        Returns:
            float: The longitudinal momentum of the particle.
        '''
        p = self.TLV.P()
        pt = self.TLV.Pt()
        return TMath.Sqrt((p - pt) * (p + pt))

    @property
    def eta(self) -> float:
        '''Returns the pseudorapidity (Eta) of the Particle object.

        Returns:
            float: The pseudorapidity (Eta) of the particle.
        '''
        TLV = self.TLV
        return TLV.Eta()

    @property
    def phi(self) -> float:
        '''Returns the azimuthal angle (Phi) of the Particle object.

        Returns:
            float: The azimuthal angle (Phi) of the particle.
        '''
        phi = self.TLV.Phi()
        return phi

    @property
    def m(self) -> float:
        '''Returns the reconstructed mass (M) of the Particle object.

        Returns:
            float: The reconstructed mass (M) of the particle.
        '''
        TLV = self.TLV
        return TLV.M()

    @property
    def energy(self) -> float:
        '''Returns the reconstructed energy of the Particle object.

        Returns:
            float: The reconstructed energy of the particle.
        '''
        TLV = self.TLV
        return TLV.Energy()
    
    def set_good_tag(self, value):
        ''' Sets the GoodTag label of the particle.
        
        Parameters:
            value (int): Goodtag to be set. It should be 0 or 1.
        '''
        if value not in [0, 1]:
            raise ValueError("Error: GoodTag value should be 0 or 1.")
        self.GoodTag = value
    
    def get_good_tag(self,cuts):
        ''' Define and returns a label (GoodTag) that indicate if particle is within the range of kinematic cuts (Pt_min, Pt_max, Eta_min, Eta_max). 
        
        Goodtag could have two values: 
        1: Particle is within the range of kinematic cuts.
        0: Particle is not within the range of kinematic cuts.
        
        Parameters:
            cuts (dict): contains the values of kinematic cuts. It should have the keys "pt_min_cut", "pt_max_cut" (optional), "eta_min_cut" and "eta_max_cut".
        
        Returns: 
            float: Goodtag.
        '''
        kin_cuts=cuts.get(self.Type)
        
        pt_min_cut=kin_cuts.get("pt_min_cut")
        pt_max_cut=kin_cuts.get("pt_max_cut")#optional
        eta_min_cut=kin_cuts.get("eta_min_cut")
        eta_max_cut=kin_cuts.get("eta_max_cut")
        
        pt_cond= (self.pt>= pt_min_cut)
        if pt_max_cut:
            if not (pt_max_cut>pt_min_cut):
                raise Exception("Error: pt_max must be major than pt_min")
            pt_cond = pt_cond and (self.pt<= pt_max_cut)
        eta_cond = (self.eta>= eta_min_cut) and (self.eta<= eta_max_cut)
        
        if (pt_cond and eta_cond):
            self.set_good_tag(1)
        else:
            self.set_good_tag(0)
            
        return self.GoodTag
    
    ### Delta methods
    def DeltaR(self, v2):
        ''' Calculates and returns DeltaR between particle (self) and other object of particle class (v2).
        
        Parameters:
            v2 (Particle): Another particle to calculate DeltaR respect to the main particle (self).
        
        Returns:
            float: DeltaR.
        '''
        TLV1 = self.TLV
        TLV2 = v2.GetTLV()
        return TLV1.DeltaR(TLV2)

    def DeltaEta(self, v2):
        ''' Calculates and returns DeltaEta between particle (self) and other object of particle class (v2).
        
        Parameters:
            v2 (Particle: Another particle to calculate DeltaEta respect to the main particle (self).
        
        Returns:
            float: DeltaEta.
        '''
        TLV1 = self.TLV
        TLV2 = v2.GetTLV()
        return (TLV1.Eta() - TLV2.Eta())

    def DeltaPhi(self, v2):
        ''' Calculates and returns DeltaPhi between particle (self) and other object of particle class (v2).
        
        Parameters:
            v2 (Particle): Another particle to calculate DeltaPhi respect to the main particle (self).
        
        Returns:
            float: DeltaPhi.
        '''
        TLV1 = self.TLV
        TLV2 = v2.GetTLV()
        return TLV1.DeltaPhi(TLV2)

    def sDeltaPT(self, v2):
        ''' Calculates and returns sDeltaPT (s - scalar) between particle (self) and other object of particle class (v2).
        
        Parameters:
            v2 (Particle): Another particle to calculate sDeltaPT respect to the main particle (self).
        
        Returns:
            float: sDeltaPT.
        '''
        TLV1 = self.TLV
        TLV2 = v2.GetTLV()
        return (TLV1.Pt() - TLV2.Pt())

    def vDeltaPT(self, v2):
        ''' Calculates and returns vDeltaPT (v - vectorial) between particle (self) and other object of particle class (v2).
        
        Parameters:
            v2 (Particle): Another particle to calculate vDeltaPT respect to the main particle (self).
        
        Returns:
            float: vDeltaPT.
        '''      
        TLV1 = self.TLV
        TLV2 = v2.GetTLV()
        a=TVector2(TLV1.Px(), TLV1.Py())
        b=TVector2(TLV2.Px(), TLV2.Py())
        c=a-b
        return c.Mod()

    def vDeltaP(self,v2):
        ''' Calculates and returns vDeltaP (v - vectorial) between particle (self) and other object of particle class (v2).
        
        Parameters:
            v2 (Particle): Another particle to calculate vDeltaP respect to the main particle (self).
        
        Returns:
            float: vDeltaP.
        '''          
        TLV1 = self.TLV
        TLV2 = v2.GetTLV()
        a=TVector3(TLV1.Px(), TLV1.Py(), TLV1.Pz())
        b=TVector3(TLV2.Px(), TLV2.Py(), TLV2.Pz())
        c=a-b
        return c.Mag()