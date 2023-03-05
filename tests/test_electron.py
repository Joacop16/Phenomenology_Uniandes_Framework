import pytest,os
import ROOT
from Uniandes_Framework.delphes_reader.particle.electron_particle import ElectronParticle

@pytest.fixture()
def event():
    tree = ROOT.TChain("Delphes;1") 
    tree.Add(os.path.join("Uniandes_Framework","tests","data","delphes_test.root"))
    for event in tree:
        return event
        break

def test_electron_particle(event):
    electron = ElectronParticle(event, 0)
    assert electron.Type == "electron"
    assert electron.Name == "e"
    assert electron.Charge == 1
    assert electron.p == pytest.approx(187.92, rel=1e-2)
