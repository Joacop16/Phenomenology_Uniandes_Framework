import pytest
import ROOT
import os
from Uniandes_Framework.delphes_reader.classifier import  DEFAULT_CUTS, get_met, get_muons, get_electrons, get_unified, get_leptons, get_jets, get_good_particles, get_good_jets, get_good_leptons
from Uniandes_Framework.delphes_reader.particle import MetParticle, MuonParticle, ElectronParticle
from Uniandes_Framework.delphes_reader import Quiet 

# Define a fixture that loads a Delphes file for testing
@pytest.fixture
def event():
    with Quiet():
        tree = ROOT.TChain("Delphes")
        tree.Add(os.path.join(os.getcwd(),"tests","data","delphes_test.root"))
        return next(iter(tree))

@pytest.fixture
def tree():
    with Quiet():
        tree = ROOT.TChain("Delphes")
        tree.Add(os.path.join(os.getcwd(),"tests","data","delphes_test.root"))
        return tree

def test_get_met(event):
    # Test that get_met() returns a MetParticle
    met = get_met(event)
    assert isinstance(met, MetParticle)

def test_get_muons(event):
    # Test that get_muons() returns a list of MuonParticles
    muons = get_muons(event)
    assert isinstance(muons, list)
    assert all(isinstance(muon, MuonParticle) for muon in muons)

    # Test that the muons are sorted by pt
    assert all(muons[i].pt >= muons[i+1].pt for i in range(len(muons)-1))

    # Check that the number of muons is correct
    assert len(muons) == event.Muon.GetEntries()

def test_get_electrons(event):
    # Test that get_electrons() returns a list of ElectronParticles
    electrons = get_electrons(event)
    assert isinstance(electrons, list)
    assert all(isinstance(electron, ElectronParticle) for electron in electrons)

    # Test that the electrons are sorted by pt
    assert all(electrons[i].pt >= electrons[i+1].pt for i in range(len(electrons)-1))

    # Check that the number of electrons is correct
    assert len(electrons) == event.Electron.GetEntries()

def test_get_unified(event):
    # Test that get_unified() returns a dictionary with only one key, "all"
    muons = get_muons(event)
    electrons = get_electrons(event)
    unified = get_unified({"muon": muons, "electron": electrons})
    assert set(unified.keys()) == set(["all"])

    # Test that the unified list is sorted by pt
    assert all(unified["all"][i].pt >= unified["all"][i+1].pt for i in range(len(unified["all"])-1))

    # Check that the number of particles is correct
    assert len(unified["all"]) == event.Muon.GetEntries() + event.Electron.GetEntries()

    # Check that the unified list contains all muons and electrons
    assert all(p in unified["all"] for p in muons)
    assert all(p in unified["all"] for p in electrons)

def test_get_leptons(event):
    # Test that get_leptons() returns a dictionary with keys "muon" and "electron"
    leptons = get_leptons(event)
    assert set(leptons.keys()) == set(["muon", "electron"])

    # check that the number of leptons is correct
    assert len(get_unified(leptons)["all"]) == event.Muon.GetEntries() + event.Electron.GetEntries()

def test_get_jets(event):
    # Test that get_jets() returns a dictionary with keys "l_jet", "b_jet", "tau_jet", and "other_jet"
    jets = get_jets(event)
    assert set(jets.keys()) == set(["l_jet", "b_jet", "tau_jet", "other_jet"])

    # check that the number of jets is correct
    assert len(get_unified(jets)["all"]) == event.Jet.GetEntries()

def pretest_get_good_particles(event, kinematic_cuts=None):
    # Test that get_good_particles() returns a dictionary with keys "muon", "electron", "l_jet", "b_jet", "tau_jet", and "other_jet"
    muons = get_muons(event)
    electrons = get_electrons(event)
    jets = get_jets(event)

    good_particles = get_good_particles(
        {
            "muon": muons,
            "electron": electrons, 
            "l_jet": jets["l_jet"], 
            "b_jet": jets["b_jet"], 
            "tau_jet": jets["tau_jet"], 
            "other_jet": jets["other_jet"]
        },
        kinematic_cuts
    )

    if kinematic_cuts is None:
        kinematic_cuts = DEFAULT_CUTS

    part_keys = ["muon", "electron", "l_jet", "b_jet", "tau_jet", "other_jet"]

    # check that the number of good particles is correct
    assert set(good_particles.keys()) == set(part_keys)
    

    for key in part_keys:
        # check that kinematic pt min cuts are applied
        assert all(p.pt > kinematic_cuts[key].get("pt_min_cut") for p in good_particles[key])

        # check that kinematic pt max optional cuts are applied
        assert all(p.pt < kinematic_cuts[key].get("pt_max_cut", 1e10) for p in good_particles[key])

        # check that kinematic eta min cuts are applied
        assert all(abs(p.eta) > kinematic_cuts[key].get("eta_min_cut") for p in good_particles[key])

        # check that kinematic eta max cuts are applied
        assert all(abs(p.eta) < kinematic_cuts[key].get("eta_max_cut") for p in good_particles[key])

    # Check that all good particles are separated at least by dR = 0.3
    all_good_particles = get_unified(good_particles)["all"]
    for i in range(len(all_good_particles)):
        for j in range(i+1, len(all_good_particles)):
            assert all_good_particles[i].DeltaR(all_good_particles[j]) >= 0.3

def test_get_good_particles_tree(tree):
    # Check get_good_particles() with default kinematic cuts
    list(map(pretest_get_good_particles,tree, [None]*tree.GetEntries()))

    # Check get_good_particles() with custom kinematic cuts
    cut = { "pt_min_cut": 100., "pt_max_cut": 500., "eta_min_cut": -1.8, "eta_max_cut": +1.8 }
    kinematic_cuts = {
        "muon": cut,
        "electron": cut,
        "l_jet": cut,
        "b_jet": cut,
        "tau_jet": cut,
        "other_jet": cut
    }
    list(map(pretest_get_good_particles,tree, [kinematic_cuts]*tree.GetEntries()))

def test_get_good_leptons(tree):
    for event in tree: 
        good_leptons = get_good_leptons(event)

        # Check name of leptons
        assert all( lepton.Name.startswith("lep_") for lepton in good_leptons)

        # Check that have a unique Name
        assert len(set(lepton.Name for lepton in good_leptons)) == len(good_leptons)

        # Check that are sorted by pt
        assert all(good_leptons[i].pt >= good_leptons[i+1].pt for i in range(len(good_leptons)-1))

        # Check that name is sorted by pt
        assert all(good_leptons[i].Name < good_leptons[i+1].Name for i in range(len(good_leptons)-1))

        # Check that comes from electrons or muons
        assert all(lepton.Type == "electron" or lepton.Type == "muon" for lepton in good_leptons)

def test_get_good_jets(tree):
    for event in tree: 
        good_jets = get_good_jets(event)
        for key in good_jets.keys():
            # Check name of jets
            assert all( jet.Name.startswith(key+"_") for jet in good_jets[key])

            # Check that have a unique Name
            assert len(set(jet.Name for jet in good_jets[key])) == len(good_jets[key])

            # Check that are sorted by pt
            assert all(good_jets[key][i].pt >= good_jets[key][i+1].pt for i in range(len(good_jets[key])-1))

            # Check that name is sorted by pt
            assert all(good_jets[key][i].Name < good_jets[key][i+1].Name for i in range(len(good_jets[key])-1))

            # Check that comes from electrons or muons
            assert all(jet.Type == key for jet in good_jets[key])