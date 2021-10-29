import sys
import os
path = os.path.join(os.path.dirname(__file__), '../../')
if path not in sys.path:
    sys.path.append(path)

import PassiveWFMeasurement.wf_model as wf_model


struct = wf_model.get_structure('SARUN18-UDCP010')
struct2 = wf_model.get_structure('SARUN18-UDCP010')


print(struct == struct2)
{struct: 0}


