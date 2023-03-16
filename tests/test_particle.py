import pytest,os
import ROOT
from ROOT import TLorentzVector
from Uniandes_Framework.delphes_reader.particle.abstract_particle import Particle
from Uniandes_Framework.delphes_reader.particle.electron_particle import ElectronParticle
from Uniandes_Framework.delphes_reader.particle.jet_particle import JetParticle
from Uniandes_Framework.delphes_reader.particle.met_particle import MetParticle
from Uniandes_Framework.delphes_reader import Quiet 


@pytest.fixture
def particle():
    return Particle(charge=-1.0, name="e-", particle_type="electron")

def test_particle_creation(particle):
    assert isinstance(particle, Particle)
    assert isinstance(particle.TLV, TLorentzVector)
    assert particle.Charge == -1.0
    assert particle.Name == "e-"
    assert particle.Type == "electron"

def test_pt(particle):
    particle.TLV.SetPtEtaPhiM(50, 1.5, 0.5, 0.5)
    assert particle.pt == 50

def test_p(particle):
    particle.TLV.SetPtEtaPhiM(5, 0.97545, 0.95055, 0.5)
    assert particle.p == pytest.approx(7.5734, rel=1e-5)

def test_pl(particle):
    particle.TLV.SetPtEtaPhiM(5, 0.97545, 0.95055, 0.5)
    assert particle.pl == pytest.approx(5.6883, rel=1e-5)

def test_eta(particle):
    particle.TLV.SetPtEtaPhiM(5, 0.97545, 0.95055, 0.5)
    assert particle.eta == pytest.approx(0.97545, rel=1e-5)

def test_phi(particle):
    particle.TLV.SetPtEtaPhiM(5, 0.97545, 0.95055, 0.5)
    assert particle.phi == pytest.approx(0.95055, rel=1e-5)

def test_m(particle):
    particle.TLV.SetPtEtaPhiM(5, 0.97545, 0.95055, 0.5)
    assert particle.m == pytest.approx(0.5, rel=1e-5)

def test_energy(particle):
    particle.TLV.SetPtEtaPhiM(5, 0.97545, 0.95055, 0.5)
    assert particle.energy == pytest.approx(7.5899, rel=1e-5)

def test_get_charge(particle):
    assert particle.GetCharge() == -1.0

def test_get_name(particle):
    assert particle.GetName() == "e-"

def test_set_name(particle):
    particle.SetName("positron")
    assert particle.Name == "positron"
    
def test_delta_eta(particle):
    particle1 = Particle(charge=1.0, name="p", particle_type="proton")
    particle1.TLV.SetPtEtaPhiM(5, 0.97545, 0.95055, 0.5)
    particle2 = Particle(charge=-1.0, name="p", particle_type="proton")
    particle2.TLV.SetPtEtaPhiM(5, 1.3316, -2.19199, 0.5)

    assert particle1.DeltaEta(particle2) == pytest.approx(-0.35615, rel=1e-5)
    assert particle2.DeltaEta(particle1) == pytest.approx(0.35615, rel=1e-5)

def test_delta_phi(particle):
    particle1 = Particle(charge=1.0, name="p", particle_type="proton")
    particle1.TLV.SetPtEtaPhiM(5, 0.97545, 0.95055, 0.5)
    particle2 = Particle(charge=-1.0, name="p", particle_type="proton")
    particle2.TLV.SetPtEtaPhiM(5, -0.35615, -2.19199, 0.5)

    assert particle1.DeltaPhi(particle2) == pytest.approx(-3.14064, rel=1e-5)
    assert particle2.DeltaPhi(particle1) == pytest.approx(3.14064, rel=1e-5)

def test_delta_r(particle):
    particle1 = Particle(charge=1.0, name="p", particle_type="proton")
    particle1.TLV.SetPtEtaPhiM(5.09902, 1.236, 0.876, 0.5)
    particle2 = Particle(charge=-1.0, name="p", particle_type="proton")
    particle2.TLV.SetPtEtaPhiM(5.09902, -1.236, -2.265, 0.5)

    assert particle1.DeltaR(particle2) == pytest.approx(3.99708, rel=1e-5)
    assert particle2.DeltaR(particle1) == pytest.approx(3.99708, rel=1e-5)


def test_tlv():
    tlv = TLorentzVector(1,2,3,4)
    assert isinstance(tlv, TLorentzVector)
    

@pytest.fixture
def event():
    with Quiet():
        tree = ROOT.TChain("Delphes")
        tree.Add(os.path.join("Uniandes_Framework","tests","data","delphes_test.root"))
        return next(iter(tree))

# Define a test function to test the ElectronParticle class
def test_electron_particle(event):
    # Initialize an ElectronParticle object
    electron = ElectronParticle(event, 0)
    
    # Test the TLV attribute
    assert isinstance(electron.TLV, ROOT.TLorentzVector)
    assert electron.p == pytest.approx(187.9219, rel = 1e-3)
    assert electron.pt == pytest.approx(187.90409, rel = 1e-3)             
    assert electron.eta == pytest.approx(-0.01377586, rel = 1e-3)
    assert electron.phi == pytest.approx(2.29576, rel = 1e-3)
    assert electron.m == pytest.approx(0.000511, rel = 1e-3)
    assert electron.energy == pytest.approx(187.9219, rel = 1e-3)
    # Test the Charge attribute
    assert electron.Charge == 1
    
    # Test the Name attribute
    assert electron.Name == "e"
    
    # Test the Type attribute
    assert electron.Type == "electron"



def test_jet_particle_initialization(event):
    # Create a JetParticle object
    jet = JetParticle(event, 0)
    # Check that the object attributes were initialized correctly
    assert jet.Type == 'l_jet'
    assert jet.Name == 'l_jet_{0}'
    assert jet.BTag == 0 or jet.BTag == 1
    assert jet.TauTag == 0 or jet.TauTag == 1
    assert jet.CTag == 0 or jet.CTag == 1
    assert jet.Flavor in [0,1,2,3,4,5,22]
    assert isinstance(jet.Charge, float)
    
    # Test the TLV attribute
    assert isinstance(jet.TLV, ROOT.TLorentzVector)
    assert jet.p == pytest.approx(127.211, rel = 1e-3)
    assert jet.pt == pytest.approx(122.89, rel = 1e-3)             
    assert jet.eta == pytest.approx(-0.26433, rel = 1e-3)
    assert jet.phi == pytest.approx(-1.5713, rel = 1e-3)
    assert jet.m == pytest.approx(4.2627, rel = 1e-3)
    assert jet.energy == pytest.approx(127.211, rel = 1e-3)
    

    
# test_met_particle.py
def test_met_particle_init(event):
    met_particle = MetParticle(event)
    assert isinstance(met_particle, MetParticle)
    assert met_particle.Charge == 0.0
    assert met_particle.Name == "MET"
    assert met_particle.Type == "MET"
    # Test the TLV attribute
    assert isinstance(met_particle.TLV, ROOT.TLorentzVector)
    assert met_particle.p == pytest.approx(69.636, rel = 1e-3)
    assert met_particle.pt == pytest.approx(69.636, rel = 1e-3)
    assert met_particle.energy == pytest.approx(69.636, rel = 1e-3)
    assert met_particle.phi == pytest.approx(0.45084, rel = 1e-3)
    assert met_particle.eta == pytest.approx(0.0, abs = 1e-3)
    assert met_particle.pl == pytest.approx(0.0, abs = 1e-3)           
    assert met_particle.m == pytest.approx(0.0, abs = 1e-3)

