import pytest
import os
import csv
import shutil
import tempfile
from Uniandes_Framework.delphes_reader.loader import DelphesLoader
from Uniandes_Framework.delphes_reader import Quiet 

@pytest.fixture(scope="session")
def test_data_dir():
    # create a temporary directory that will contain the test data
    tmp_dir = tempfile.mkdtemp()

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))

    # copy the contents of the data directory to the temporary directory
    shutil.copytree(data_path, os.path.join(tmp_dir, "data"))

    # Create a csv file with the paths to the root files
    with open(os.path.join(tmp_dir, "example_paths.csv"), "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["signal", "path", "xs"])
        writer.writerow(["example_signal", os.path.join(tmp_dir, "data"), 100])

    # return the temporary directory as a fixture
    yield tmp_dir

    # remove the directory and its contents
    shutil.rmtree(tmp_dir)

def test_load_root_file(test_data_dir):
    # load the root file
    loader = DelphesLoader("example_signal", os.path.join(test_data_dir, "example_paths.csv"))

    # check the name
    assert loader.name == "example_signal"

    # check the cross section
    assert loader.xs == "100"
    
    # check the path
    assert loader._path_to_signal == os.path.join(test_data_dir, "data")

    # check the path to the root file
    assert loader.Forest == [os.path.join(test_data_dir, "data", "delphes_test.root")]

    # check the number of events
    with Quiet():
        assert loader.get_nevents() == 500

def test_invalid_signal():
    with pytest.raises(Exception):
        DelphesLoader("invalid_signal")

def test_default_path():
    loader = DelphesLoader("ttbar")
    
    # Check the path to the ttbar location
    assert loader._path_to_signal == "/Madgraph_Simulations/SM_Backgrounds/ttbar/"
    
    # Check the cross section of ttbar
    assert float(loader.xs) == pytest.approx(504, abs=1)

def test_read_path(test_data_dir):
    loader = DelphesLoader("example_signal", os.path.join(test_data_dir, "example_paths.csv"))

    # Check the path to the root file
    expected_dict = {"signal": ["path", "xs"], "example_signal": [os.path.join(test_data_dir, "data"), "100"]}
    assert loader._read_path(os.path.join(test_data_dir, "example_paths.csv")) == expected_dict

def test_missing_data_file():
    with pytest.raises(Exception):
        DelphesLoader("example_signal", "invalid_file.csv")

def test_set_glob(test_data_dir):
    loader = DelphesLoader("example_signal", os.path.join(test_data_dir, "example_paths.csv"))
    loader.set_glob("asd")

    # Check the glob when it is defined and that have been set
    assert loader._glob == "asd"

def test_get_glob(test_data_dir):
    loader = DelphesLoader("example_signal", os.path.join(test_data_dir, "example_paths.csv"))
    # Check the default glob
    assert loader.get_glob() == "**/*.root"

    # Check the glob when it is defined
    loader.set_glob("asd")
    assert loader.get_glob() == "asd"

def test_get_nevents(test_data_dir):
    loader = DelphesLoader("example_signal", os.path.join(test_data_dir, "example_paths.csv"))
    # Check the number of events
    with Quiet():
        assert loader.get_nevents() == 500

