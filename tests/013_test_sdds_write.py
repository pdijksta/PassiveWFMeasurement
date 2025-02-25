import numpy as np
from PassiveWFMeasurement import wf_model

structure = wf_model.get_structure('SARUN18-UDCP020')

tt = np.linspace(0, 100e-15, int(1e4))

structure.generate_elegant_wf('./test_wake.sdds', tt, 10e-3, 4.7e-3)


