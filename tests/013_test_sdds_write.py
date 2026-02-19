import numpy as np
from PassiveWFMeasurement import wf_model

# see config.py, aramis_structure_parameters
structure = wf_model.get_structure('SARUN18-UDCP020')

# Wake function time grid
tt = np.linspace(0, 100e-15, int(1e4))

# provides sdds file with entries t WL WXD WXQ (longitudinal, transverse dipole, transverse quadrupole wake)
structure.generate_elegant_wf('./test_wake.sdds', tt, 10e-3, 4.7e-3)

