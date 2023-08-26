import base64
import itertools
import time
import numpy as np
import logging
import datetime

import pyscan
from cam_server import PipelineClient
from cam_server.utils import get_host_port_from_stream_address
from epics import caget, caput

from . import config

def get_readables(beamline):
    return [
            'bs://image',
            'bs://x_axis',
            'bs://y_axis',
            config.beamline_chargepv[beamline],
            ]

def pyscan_result_to_dict(readables, result, scrap_bs=False):
    """
    Excpects a nested list of order 3.
    Level 1 is the scan index.
    Level 2 is the number of images per scan index (unless this number is 1 in which case this level does not exist).
    Level 3 is the number of readables.

    Returns a shuffled version that takes the form of the dictionary, with the readables as keys.
    """

    output = {}

    for nR, readable in enumerate(readables):
        readable_output1 = []
        for level_scan in result:
            readable_output2 = []
            for level_image in level_scan:
                readable_output2.append(level_image[nR])
            readable_output1.append(readable_output2)

        if scrap_bs and hasattr(readable, 'startswith') and readable.startswith('bs://'):
            readable2 = readable[5:]
        else:
            readable2 = readable

        try:
            output[readable2] = np.array(readable_output1)
        except:
            output[readable2] = readable_output1

    return output

def get_undulator_K(beamline):
    pvs = config.beamline_undulators[beamline]
    return pvs, np.array([caget(pv) for pv in pvs])

def destroy_lasing(beamline, dry_run, max_deltaK=0.2):
    pvs, old_vals = get_undulator_K(beamline)
    randoms = np.random.rand(len(pvs)) - 0.5
    new_vals = old_vals + randoms * max_deltaK
    for pv, new_val in zip(pvs, new_vals):
        if dry_run:
            print('I would caput %s %.6f' % (pv, new_val))
        else:
            caput(pv, new_val)
    return pvs, old_vals, new_vals

def restore_lasing(pvs, vals, dry_run):
    for pv, val in zip(pvs, vals):
        if dry_run:
            print('I would caput %s %.6f' % (pv, val))
        else:
            caput(pv, val)

def get_images(screen, n_images, beamline, dry_run=None):
    if dry_run:
        daq_screen = 'simulation'
    else:
        daq_screen = screen
    print('Start get_images for screen %s, %i images, beamline %s' % (screen, n_images, beamline))

    def dummy_func(*args):
        pass

    meta_dict_1 = get_meta_data(screen, dry_run, beamline)
    #positioner = pyscan.BsreadPositioner(n_messages=n_images)
    positioner = pyscan.VectorPositioner([0])
    writables = [pyscan.function_value(dummy_func, 'dummy')]
    settings = pyscan.scan_settings(settling_time=0.01, measurement_interval=0.2, n_measurements=n_images)

    pipeline_client = PipelineClient('http://sf-daqsync-01:8889/')
    cam_instance_name = str(daq_screen) + '_sp1'
    stream_address = pipeline_client.get_instance_stream(cam_instance_name)
    stream_host, stream_port = get_host_port_from_stream_address(stream_address)

    try:
        bg = pipeline_client.get_latest_background(daq_screen)
        image = pipeline_client.get_background_image_bytes(bg)
        dtype = image['dtype']
        shape = image['shape']
        bytes = base64.b64decode(image['bytes'].encode())
        background = np.array(bytes, dtype=dtype).reshape(shape)
    except Exception as e:
        print(e)
        print('Error taking background')
        background = 0

    # Configure bsread
    pyscan.config.bs_default_host = stream_host
    pyscan.config.bs_default_port = stream_port

    logging.getLogger("mflow.mflow").setLevel(logging.ERROR)

    readables = get_readables(beamline)

    raw_output = pyscan.scan(positioner=positioner, readables=readables, settings=settings, writables=writables)

    # Bugged?
    # output = [[x] for x in raw_output]
    output = raw_output

    result_dict = pyscan_result_to_dict(readables, output, scrap_bs=True)

    for ax in ['x_axis', 'y_axis']:
        arr = result_dict[ax]*1e-6 # convert to m
        if len(arr.shape) == 3:
            result_dict[ax+'_m'] = arr[0,0,:]
        elif len(arr.shape) == 2:
            result_dict[ax+'_m'] = arr[0,:]
        else:
            raise ValueError('Unexpected', len(arr.shape))

    meta_dict_2 = get_meta_data(screen, dry_run, beamline)
    output_dict = {
            'pyscan_result': result_dict,
            'meta_data_begin': meta_dict_1,
            'meta_data_end': meta_dict_2,
            'background': background,
            }

    print('End get_images')
    return output_dict

def data_structure_offset(structure, offset_range, screen, n_images, dry_run, beamline):

    print('Start data_structure_offset for structure %s, screen %s, beamline %s, dry_run %s' % (structure, screen, beamline, dry_run))
    meta_dict_1 = get_meta_data(screen, dry_run, beamline)

    pipeline_client = PipelineClient('http://sf-daqsync-01:8889/')
    offset_pv = structure+':CENTER'

    current_val = caget(offset_pv+'.RBV')

    # Start from closer edge of scan
    if abs(current_val - offset_range[0]) > abs(current_val - offset_range[-1]):
        offset_range = offset_range[::-1]

    positions = offset_range * 1e3 # convert to mm
    positioner = pyscan.VectorPositioner(positions=positions.tolist())

    def dummy_func(*args):
        print('I would set %s to' % offset_pv, args)

    if dry_run:
        daq_screen = 'simulation'
        writables = [pyscan.function_value(dummy_func, 'dummy')]
    else:
        writables = [pyscan.epics_pv(pv_name=offset_pv, readback_pv_name=offset_pv+'.RBV', tolerance=0.005)]
        daq_screen = screen

    cam_instance_name = daq_screen + '_sp1'
    stream_address = pipeline_client.get_instance_stream(cam_instance_name)
    stream_host, stream_port = get_host_port_from_stream_address(stream_address)

    # Configure bsread
    pyscan.config.bs_default_host = stream_host
    pyscan.config.bs_default_port = stream_port

    logging.getLogger('mflow.mflow').setLevel(logging.ERROR)

    settings = pyscan.scan_settings(settling_time=1, n_measurements=n_images, write_timeout=60)

    readables = get_readables(beamline)

    raw_output = pyscan.scan(positioner=positioner, readables=readables, settings=settings, writables=writables)
    result_dict = pyscan_result_to_dict(readables, raw_output, scrap_bs=True)
    #import pdb; pdb.set_trace()
    for ax in ['x_axis', 'y_axis']:
        arr = result_dict[ax]*1e-6
        if len(arr.shape) == 3:
            result_dict[ax+'_m'] = arr[0][0]
        elif len(arr.shape) == 2:
            result_dict[ax+'_m'] = arr[0]
        else:
            raise ValueError('Unexpected', len(arr.shape))

    meta_dict_2 = get_meta_data(screen, dry_run, beamline)

    output = {
            'pyscan_result': result_dict,
            'streaker_offsets': offset_range,
            'screen': screen,
            'n_images': n_images,
            'dry_run': dry_run,
            'structure': structure,
            'meta_data_begin': meta_dict_1,
            'meta_data_end': meta_dict_2,
            }
    print('End data_structure_offset')
    return output

def move_pv(pv, value, timeout, tolerance):
    caput(pv, value)
    step_seconds = 0.1
    max_step = timeout // step_seconds
    for step in range(max_step):
        current_value = caget(pv+'.RBV')
        if abs(current_value - value) < tolerance:
            break
        else:
            time.sleep(step_seconds)
        if step % 10 == 0:
            caput(pv, value)
    else:
        raise ValueError('Pv %s should be %e, is: %e after %f seconds!' % (pv, value, current_value, timeout))

def get_quad_strengths(beamline):
    quads = config.beamline_quads[beamline]

    k1l_dict = {}
    for quad in quads:
        k1l_dict[quad] = caget(quad.replace('.', '-')+':K1L-SET')
    energy_pv = config.beamline_energypv[beamline]
    k1l_dict[energy_pv] = caget(energy_pv)
    return k1l_dict

def get_meta_data(screen, dry_run, beamline):
    all_structures = config.structure_names[beamline]
    meta_dict = {}
    for streaker, suffix1, suffix2 in itertools.product(all_structures, [':GAP', ':CENTER'], ['', '.RBV']):
        pv = streaker+suffix1+suffix2
        meta_dict[pv] = caget(pv)
    for bl in config.swissfel_beamlines:
        pv = config.beamline_chargepv[bl]
        meta_dict[pv] = caget(pv)
        pv = config.gas_monitor_pvs[bl]
        meta_dict[pv] = caget(pv)

    energy_pv = screen+':ENERGY-OP'
    if dry_run:
        meta_dict[energy_pv] = 6000
    else:
        meta_dict[energy_pv] = caget(energy_pv)

    k1l_dict = get_quad_strengths(beamline)
    meta_dict.update(k1l_dict)
    meta_dict['time'] = str(datetime.datetime.now())
    return meta_dict

def set_optics(quads, k1ls, dry_run):
    for quad, k1l in zip(quads, k1ls):
        if dry_run:
            print('I would caput %s %.3f' % (quad, k1l))
        else:
            caput(quad.replace('.','-')+':K1L-SET', k1l)

