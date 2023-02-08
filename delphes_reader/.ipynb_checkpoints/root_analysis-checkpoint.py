import ROOT
import pandas as pd

def get_kinematics_row(*args):
    ''' Extracts main kinematic variables of a particle (or more) and returns a dictionary with them.
    Parameters:
        *args (Particle): arguments could be many particle as it want to get its kinematics variable. For example: muons[0],muons[1] where muons is a list of muon particles.
    Return:
        Python dictionary: Contains main kinematic variables.
    '''  
    particles=list(args)
    co_particles=particles.copy()
    row = {} #{'header': value}
    for particle in particles:
        name=particle.Name
        row[f"pT_{{{name}}}(GeV)"]=particle.pt()
        row[f"#eta_{{{name}}}"]=particle.eta()
        row[f"#phi_{{{name}}}"]=particle.phi()
        row[f"Energy_{{{name}}}(GeV)"]=particle.energy()
        row[f"Mass_{{{name}}}(GeV)"]=particle.m()
        co_particles.pop(0)
        for co_particle in co_particles:
            co_name=co_particle.Name
            row[f"#Delta R_{{{name}{co_name}}}"]=particle.DeltaR(co_particle)
            row[f"#Delta #eta_{{{name}{co_name}}}"]=particle.DeltaEta(co_particle)
            row[f"#Delta #phi_{{{name}{co_name}}}"]=particle.DeltaPhi(co_particle)
            row[f"#Delta pT_{{{name}{co_name}}}(GeV)"]=particle.sDeltaPT(co_particle)
            row[f"#Delta #vec{{pT}}_{{{name}{co_name}}}(GeV)"]=particle.vDeltaPT(co_particle)
            row[f"#Delta #vec{{p}}_{{{name}{co_name}}}(GeV)"]=particle.vDeltaP(co_particle)
    return row

default_hist_bins_dict={
    "#Delta R":[96,0,7],
    "#Delta #eta":[80,-5,5],
    "#Delta #phi":[52,-3.25,3.25],
    "#Delta pT":[120, 0.0, 1500.0],
    "#Delta #vec{pT}":[240, 0.0, 4800.0],
    "#Delta #vec{p}":[240, 0.0, 4800.0],
    "MET(GeV)":[80, 0.0, 1000.0],
    "pT_": [160, 0.0, 2000.0],
    "sT(GeV)": [200, 0.0, 4000.0],
    "mT(GeV)": [200, 0.0, 4000.0],
    "#eta_":[80, -5, 5],
    "#phi_":[128, -3.2, 3.2],
    "Energy_":[80, 0.0, 1000.0]
}

def make_histograms(df,integral=1.0,hist_bins_dict=default_hist_bins_dict):
    ''' Uses ROOT to create histograms using all data contained in a DataFrame.  
    Parameters:
        df (DataFrame): It is a DataFrame where each row correspond to a different particle and each column to its corresponding kinematic variable value.
    Return:
        TH1F (Python dictionary): Contains all histograms. The keys of this dictionary are kinematic variable names (the same keys of the DataFrame).
    '''  
    hist_dict={}
    for key in dict(df).keys():
        for hist_key in hist_bins_dict.keys():
            if hist_key in key:
                bins=hist_bins_dict.get(hist_key,False)
                break
        if not (bins): 
            print("No Histogram Dictionary for ",key)
        else: 
            x_axis = key.replace('(' , "[" ).replace(')' , "]" )
            h=ROOT.TH1F(
                key, f"; {x_axis}; Events",
                bins[0],bins[1],bins[2]
            )
            h.SetDirectory(0)
            for value in df[key]:
                h.Fill(value)
            try: 
                h.Scale(integral/h.Integral())
            except:
                print(h.GetName(), " is empty!")
            hist_dict.update({key : h})
    return hist_dict

def generate_csv(directory_list,file_name):
    ''' Uses Pandas to create a csv file using all data contained in a list of directories.  
    Parameters:
        directory_list (Python list): It is a list where each member is a directory with the structure of get_kinematics_row outputs.
    '''      
    Data = pd.DataFrame()
    for directory_kinematics in directory_list:
        row = pd.DataFrame.from_dict(directory_kinematics, orient = "index").T
        Data = pd.concat([Data,row]) 
        Data.reset_index(drop=True, inplace=True)
    Data.to_csv(file_name, index= False)

class Quiet:
    ''' Context manager for silencing certain ROOT operations.  Usage:
    with Quiet(level = ROOT.kInfo+1):
       foo_that_makes_output

    You can set a higher or lower warning level to ignore different
    kinds of messages.  After the end of indentation, the level is set
    back to what it was previously.
    '''
    def __init__(self, level=ROOT.kError+1):
        self.level = level

    def __enter__(self):
        self.oldlevel = ROOT.gErrorIgnoreLevel
        ROOT.gErrorIgnoreLevel = self.level

    def __exit__(self, type, value, traceback):
        ROOT.gErrorIgnoreLevel = self.oldlevel