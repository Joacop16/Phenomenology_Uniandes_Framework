from abc import ABC
from ROOT import (
    TLorentzVector,
    TVector3,
    TVector2,
    TMath
)

class Particle(ABC):
    ''' Class Particle: 
    
        Attributes:

        TLV: TLorentzVector that is definning using Pt, Eta, Phi and M.
        Charge: Particle charge.
        Name: Particle Name (examples: #mu, e, MET, etc.).
        Type: Particle type (examples: muon, electron, MET, etc.).
    '''    
    def __init__(self,*args,**kwargs):
        ''' Initialize the Particle from the kwargs.'''
        self.TLV=TLorentzVector()
        self.Charge=kwargs.get("charge",0.0)
        self.Name=""
        self.Type=""
    #
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
    
    def SetName (self,newName):
        ''' Change name (attribute) of particle. It does not return anything.'''
        self.Name=newName
    
    def pt(self):
        ''' Returns tranverse momentum (Pt) of particle.
        
        Returns:
            float: Tranverse momentum (Pt).
        '''
        TLV = self.TLV
        return TLV.Pt()
    
    def p(self):
        ''' Returns full momentum (P) of particle.
        
        Returns:
            float: Full momentum (P).
        '''
        TLV = self.TLV
        return TLV.P()
    
    def pl(self):
        ''' Returns longitudinal momentum of particle. It is estimated using full momentum (P) and transverse momentum (Pt).
        
        Returns:
            float: Longitudinal momentum.
        '''
        p = self.TLV.P()
        pt= self.TLV.Pt()
        return TMath.Sqrt((p-pt)*(p+pt))

    def eta(self):
        ''' Returns particle pseudorapidity (Eta) of particle.
        
        Returns:
            float: Pseudorapidity (Eta).
        '''
        TLV = self.TLV
        return TLV.Eta()

    def phi(self):
        ''' Returns azimutal angle (Phi) of particle.
        
        Returns:
            float: Azimutal angle (Phi).
        '''
        phi=self.TLV.Phi()
        return phi
    
    def m(self):
        ''' Returns reconstructed particle mass (M) of particle.
        
        Returns:
            float: Reconstructed mass (M).
        '''
        TLV = self.TLV
        return TLV.M()
    
    def energy(self):
        ''' Returns reconstructed particle energy of particle.
        
        Returns:
            float: Reconstructed energy.
        '''
        TLV = self.TLV
        return TLV.Energy()

    def get_good_tag(self,cuts):
        ''' Define and returns a label (GoodTag) that indicate if particle is within the range of kinematic cuts (Pt_min, Pt_max, Eta_min, Eta_max). 
        
        Goodtag could have two values: 
        1: Particle is within the range of kinematic cuts.
        0: Particle is not within the range of kinematic cuts.
        
        Parameters:
            cuts (python dictionary): contains the values of kinematic cuts. It should have the keys "pt_min_cut", "pt_max_cut" (optional), "eta_min_cut" and "eta_max_cut".
        
        Returns: 
            float: Goodtag.
        '''
        jet_cuts=cuts.get(self.Type)
        
        pt_min_cut=jet_cuts.get("pt_min_cut")
        pt_max_cut=jet_cuts.get("pt_max_cut")#optional
        eta_min_cut=jet_cuts.get("eta_min_cut")
        eta_max_cut=jet_cuts.get("eta_max_cut")
        
        pt_cond= (self.pt()>= pt_min_cut)
        if pt_max_cut:
            if not (pt_max_cut>pt_min_cut):
                raise Exception("Error: pt_max must be major than pt_min")
            pt_cond = pt_cond and (self.pt()<= pt_max_cut)
        eta_cond = (self.eta()>= eta_min_cut) and (self.eta()<= eta_max_cut)
        
        if (pt_cond and eta_cond):
            self.GoodTag=1
        else:
            self.GoodTag=0
            
        return self.GoodTag
    
    ### Delta methods
    def DeltaR(self, v2):
        ''' Calculates and returns DeltaR between particle (self) and other object of particle class (v2).
        
        Parameters:
            v2 (Particle class object): Another particle to calculate DeltaR respect to the main particle (self).
        
        Returns:
            float: DeltaR.
        '''
        TLV1 = self.TLV
        TLV2 = v2.GetTLV()
        return TLV1.DeltaR(TLV2)

    def DeltaEta(self, v2):
        ''' Calculates and returns DeltaEta between particle (self) and other object of particle class (v2).
        
        Parameters:
            v2 (Particle class object): Another particle to calculate DeltaEta respect to the main particle (self).
        
        Returns:
            float: DeltaEta.
        '''
        TLV1 = self.TLV
        TLV2 = v2.GetTLV()
        return (TLV1.Eta() - TLV2.Eta())

    def DeltaPhi(self, v2):
        ''' Calculates and returns DeltaPhi between particle (self) and other object of particle class (v2).
        
        Parameters:
            v2 (Particle class object): Another particle to calculate DeltaPhi respect to the main particle (self).
        
        Returns:
            float: DeltaPhi.
        '''
        TLV1 = self.TLV
        TLV2 = v2.GetTLV()
        return TLV1.DeltaPhi(TLV2)

    def sDeltaPT(self, v2):
        ''' Calculates and returns sDeltaPT (s - scalar) between particle (self) and other object of particle class (v2).
        
        Parameters:
            v2 (Particle class object): Another particle to calculate sDeltaPT respect to the main particle (self).
        
        Returns:
            float: sDeltaPT.
        '''
        TLV1 = self.TLV
        TLV2 = v2.GetTLV()
        return (TLV1.Pt() - TLV2.Pt())

    def vDeltaPT(self, v2):
        ''' Calculates and returns vDeltaPT (v - vectorial) between particle (self) and other object of particle class (v2).
        
        Parameters:
            v2 (Particle class object): Another particle to calculate vDeltaPT respect to the main particle (self).
        
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
            v2 (Particle class object): Another particle to calculate vDeltaP respect to the main particle (self).
        
        Returns:
            float: vDeltaP.
        '''          
        TLV1 = self.TLV
        TLV2 = v2.GetTLV()
        a=TVector3(TLV1.Px(), TLV1.Py(), TLV1.Pz())
        b=TVector3(TLV2.Px(), TLV2.Py(), TLV2.Pz())
        c=a-b
        return c.Mag()