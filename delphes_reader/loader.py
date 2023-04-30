import os
import csv
from pathlib import Path
from ROOT import TChain
from urllib.request import urlopen

URL_DEF_PATHS = "https://raw.githubusercontent.com/Phenomenology-group-uniandes/Uniandes_Framework/main/SimulationsPaths.csv"

class DelphesLoader():
    """
    Class to load the delphes root outputs
    """

    # Constructor
    def __init__(self, name_signal, path=None):
        """
        Parameters
        ----------
        name_signal : str
            Name of the signal to load
        path : str, optional
            Path to the simulation root outputs, by default URL_DEF_PATHS
        """

        # Name of the signal
        self.name = name_signal
        # Path to the simulation root outputs
        data = self._read_path(path if path else URL_DEF_PATHS)
        
        # verify dictionary path
        try:
            self._path_to_signal = data[self.name][0] # Path
        except KeyError:
            raise Exception(f"Error: {self.name} Signal not defined")
        
        # Extract Cross Section
        self.xs = data[self.name][1]
        
        # Get the delphes root outputs
        self.Forest = self._get_forest()
        
        load=self.name+" imported with "
        load+=str(len(self.Forest)) + " trees!\n"
        load+=self._path_to_signal
        print(load, flush=True)

    # path reader to simulation root outputs
    def _read_path(self, path):
        """
        Read the path to the simulation root outputs

        parameters
        ----------
        path : str
            Path to the simulation root outputs

        returns
        -------
        dict
            Dictionary with the path to the simulation root outputs
        """
        url_protocols = ["http://", "https://", "ftp://", "ftps://" ]
        
        if any(path.startswith(p) for p in url_protocols):
            f = urlopen(path)
            reader = csv.reader(f.read().decode('utf-8').splitlines())
        else:
            if not os.path.exists(path):
                raise Exception(f"Error: {path} not found")
            f = open(path, 'r')
            reader = csv.reader(f.read().splitlines())

        data = {row[0]: row[1:] for row in reader if len(row) > 0}
        f.close()
        return data
    
    # Set and get glob to search the delphes root outputs
    def set_glob(self, glob):
        """
        Set the glob to search the delphes root outputs

        parameters
        ----------
        glob : str
            Glob to search the delphes root outputs
        """
        self._glob = glob
    ##
    def get_glob(self):
        """
        Get the glob to search the delphes root outputs when glob is defined.
        if glob is not defined, set the default glob to '**/*.root' and return it

        returns
        -------
        str
            Glob to search the delphes root outputs
        """
        if not hasattr(self, "_glob"):
            self.set_glob('**/*.root')
        return self._glob

    # Get the delphes root outputs
    def _get_forest(self, glob = None):
        """
        Get the delphes root outputs

        parameters
        ----------
        glob : str, optional
            Glob to search the delphes root outputs, by default None

        returns
        -------
        list
            Ordered list with the delphes root outputs
        """

        if glob is None:
            glob = self.get_glob()
        else: 
            self.set_glob(glob)
        def natural_sort(l): 
            import re
            convert = lambda text: int(text) if text.isdigit() else text.lower()
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            return sorted(l, key=alphanum_key)
        #Search for all root files in path to signal
        path_root = Path(self._path_to_signal)
        forest = [root_file.as_posix() for root_file in path_root.glob(glob)]
        return natural_sort(forest)
    
    
    def get_nevents(self,Forest = None):
        """
        Get the number of events in the delphes root outputs when Forest isn't None.
        if Forest is None, use the default Forest and return the number of events.
        
        parameters
        ----------
        Forest : list, optional
            List with the delphes root outputs, by default None
            
        returns
        -------
        int
            Number of events in the delphes root outputs
        """
        if Forest is None:
            Forest=self.Forest
        self.nevents=0
        for i,job in enumerate(Forest):
            tree=TChain("Delphes;1")
            tree.Add(job)
            self.nevents+=tree.GetEntries()
        return self.nevents