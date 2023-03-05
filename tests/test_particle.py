import pytest
from ROOT import TLorentzVector
from Uniandes_Framework.delphes_reader.particle.abstract_particle import Particle

@pytest.fixture
def particle():
    return Particle(charge=-1.0, name="e-", particle_type="electron")

def test_particle_creation(particle):
    assert isinstance(particle, Particle)
    assert isinstance(particle.TLV, TLorentzVector)
    assert particle.Charge == -1.0
    assert particle.Name == "e-"
    assert particle.Type == "electron"

def test_get_charge(particle):
    assert particle.GetCharge() == -1.0

def test_get_name(particle):
    assert particle.GetName() == "e-"

def test_set_name(particle):
    particle.SetName("positron")
    assert particle.Name == "positron"

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
