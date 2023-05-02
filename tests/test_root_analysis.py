import ROOT
from ROOT import TH1F, TCanvas

import numpy as np
from Uniandes_Framework.delphes_reader.root_analysis import Quiet

def test_Quiet_context_manager():
    # Create a TH1F histogram and fill it with some random numbers
    hist = TH1F("hist", "Example histogram", 100, 0, 1)
    for i in range(1000):
        hist.Fill(np.random.rand())

    with Quiet(level=ROOT.kInfo+1):
        assert ROOT.gErrorIgnoreLevel == ROOT.kInfo+1
        # Plot the histogram, which would normally produce output
        canvas = TCanvas("canvas", "Example canvas", 800, 600)
        hist.Draw()
        canvas.Draw()

    assert ROOT.gErrorIgnoreLevel == Quiet(level=ROOT.kError+1).oldlevel
