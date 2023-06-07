import gzip
import numpy as np
import xml.etree.ElementTree as ET
from Uniandes_Framework.delphes_reader.loader import DelphesLoader
from Uniandes_Framework.delphes_reader.particle.abstract_particle import Particle as AbstractParticle
from ROOT import TLorentzVector , TVector2 , TVector3

def get_kinematics_row(*args):
    particles=list(args)
    co_particles=particles.copy()
    row = {} #{'header': value}
    for particle in particles:
        name=particle.Name
        row[f"pT_{{{name}}}(GeV)"]=particle.pt
        row[f"#eta_{{{name}}}"]=particle.eta
        row[f"#phi_{{{name}}}"]=particle.phi
        row[f"Energy_{{{name}}}(GeV)"]=particle.energy
        co_particles.pop(0)
        for co_particle in co_particles:
            co_name=co_particle.Name
            row[f"#Delta{{R}}_{{{name}{co_name}}}"]=particle.DeltaR(co_particle)
            row[f"#Delta{{#eta}}_{{{name}{co_name}}}"]=particle.DeltaEta(co_particle)
            row[f"#Delta{{#phi}}_{{{name}{co_name}}}"]=particle.DeltaPhi(co_particle)
            row[f"#Delta{{pT}}_{{{name}{co_name}}}(GeV)"]=particle.sDeltaPT(co_particle)
            row[f"#Delta{{#vec{{pT}}}}_{{{name}{co_name}}}(GeV)"]=particle.vDeltaPT(co_particle)
            row[f"#Delta{{#vec{{p}}}}_{{{name}{co_name}}}(GeV)"]=particle.vDeltaP(co_particle)
    return row

class LHE_Loader(DelphesLoader):
    def __init__(self, name_signal, path=None):
        super().__init__(name_signal,path, glob = '**/*.lhe.gz')
        
class Particle():
    def __init__(self,pdgid,spin,px=0,py=0,pz=0,energy=0,mass=0):
        self.pdgid=pdgid
        self.px=px
        self.py=py
        self.pz=pz
        self.energy=energy
        self.mass=mass
        self.spin=spin
        typep=abs(pdgid)
        if ((typep==2) or (typep==4) or (typep==6)):
            self.Charge=+np.sign(pdgid)*2./3.
        elif ((typep==1) or (typep==3) or (typep==5)):
            self.Charge=-np.sign(pdgid)*1./3.
        elif ((typep==11) or (typep==13) or (typep==15)):
            self.Charge=-np.sign(pdgid)
        else: 
            self.Charge=0.0
    def SetName(self,name):
        self.Name=name
    @property
    def p4(self):
        return TLorentzVector(self.px,self.py,self.pz,self.energy)
    
    @property
    def TLV(self):
        return self.p4
        
    @p4.setter
    def p4(self,value):
        self.px=value.Px()
        self.py=value.Py()
        self.pz=value.Pz()
        self.energy=value.E()
        self.mass=value.M()
           
    @property
    def p(self):
        return self.p4.P()

    @property
    def m(self):
        return self.p4.M()

    @property
    def eta(self):
        return self.p4.Eta()
    
    @property
    def pt(self):
        return self.p4.Pt()
    
    @property
    def phi(self):
        return self.p4.Phi()
    ### Delta methods
    def GetTLV (self):    
        return self.TLV
        
    def DeltaR(self, v2):
        TLV1 = self.TLV
        TLV2 = v2.GetTLV()
        return TLV1.DeltaR(TLV2)

    def DeltaEta(self, v2):
        return self.TLV.Eta() - v2.TLV.Eta()

    def DeltaPhi(self, v2):
        TLV1 = self.TLV
        TLV2 = v2.GetTLV()
        return abs(TLV1.DeltaPhi(TLV2))

    def sDeltaPT(self, v2):
        return self.TLV.Pt() - v2.TLV.Pt()

    def vDeltaPT(self, v2):
        TLV1 = self.TLV
        TLV2 = v2.GetTLV()
        a=TVector2(TLV1.Px(), TLV1.Py())
        b=TVector2(TLV2.Px(), TLV2.Py())
        c=a-b
        return c.Mod()

    def vDeltaP(self,v2):
        TLV1 = self.TLV
        TLV2 = v2.GetTLV()
        a=TVector3(TLV1.Px(), TLV1.Py(), TLV1.Pz())
        b=TVector3(TLV2.Px(), TLV2.Py(), TLV2.Pz())
        c=a-b
        return c.Mag()
        
    def getCharge(self):
        q=self.Charge
        return float(q)
    
class Event:
    def __init__(self,num_particles):
        self.num_particles=num_particles
        self.particles=[]
    
    def __addParticle__(self,particle):
        self.particles.append(particle)
        
    def getParticlesByIDs(self,idlist):
        partlist=[]
        for pdgid in idlist:
            for p in self.particles:
                if p.pdgid==pdgid:
                    partlist.append(p)
        return partlist

class LHEFData:
    def __init__(self,version):
        self.version=version
        self.events=[]
    
    def __addEvent__(self,event):
        self.events.append(event)
        
    def getParticlesByIDs(self,idlist):
        partlist=[]
        for event in self.events:
            partlist.extend(event.getParticlesByIDs(idlist))
        return partlist
        
def get_event_by_child(child):
    lines=child.text.strip().split('\n')
    event_header=lines[0].strip()
    num_part=int(event_header.split()[0].strip())
    e=Event(num_part)
    for i in range(1,num_part+1):
        part_data=lines[i].strip().split()
        if ( int(part_data[1])!=1 ): continue
        p=Particle(
            int(part_data[0]), #pdg-id
            float(part_data[12]), #spin
            float(part_data[6]), #px
            float(part_data[7]), #py
            float(part_data[8]), #pz
            float(part_data[9]), #E
            float(part_data[10]) #m
        )
        e.__addParticle__(p)
    return e
            
def readLHEF(path_to_file):
    with gzip.open(path_to_file, 'rb') as lhe_file:
        tree = ET.parse(lhe_file)
        root=tree.getroot()
        childs=[]
        for child in root:
            if(child.tag=='event'):
                childs.append(child)
    return childs