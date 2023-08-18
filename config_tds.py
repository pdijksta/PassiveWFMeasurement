default_deltaK = 0.3

beamlines = [
        'Athos post-undulator',
        ]


devices = {
        'Athos post-undulator': ['SATMA02-RTDS100', ],
        }

beam_monitors = {
        'Athos post-undulator': ['SATBD02-DSCR050', ],
        }

pv_dict = {
        'SATMA02-RTDS100': {
            'phase': 'SATMA02-RLLE-DSP:PHASE-VS',
            'voltage': 'SATMA02-RLLE-DSP:VOLT-VS',
            },
        }



