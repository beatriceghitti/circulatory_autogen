#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:45:17 2024

@author: bghi639
"""

import sys
import os
import re
import traceback
import yaml
import pandas as pd
from collections import Counter


def generate_vessel_array_fun(inp_data_dict=None, vessels_csv_abs_path=None, parse_inp_data_dict=False):

    print("Inside generate_vessel_array_fun() function...")

    root_dir = os.path.join(os.path.dirname(__file__), '../..')
    sys.path.append(os.path.join(root_dir, 'src'))

    user_inputs_dir = os.path.join(root_dir, 'user_run_files')

    from parsers.ModelParsers import CSV0DModelParser
    from generators.CVSCellMLGenerator import CVS0DCellMLGenerator
    from generators.CVSCppGenerator import CVS0DCppGenerator
    from parsers.PrimitiveParsers import YamlFileParser
    from parsers.PrimitiveParsers import CSVFileParser

    yaml_parser = YamlFileParser()
    
    if inp_data_dict==None:
        if vessels_csv_abs_path==None:
            sys.exit("ERROR :: NO inp_data_dict and NO vessels_csv_abs_path passed as input to the function. " \
            "Please check your input arguments to this function. Exiting.")
        
        parameters_csv_abs_path = vessels_csv_abs_path.replace("_vessel_array.csv", "_parameters.csv")

        idxs = [idx for idx, char in enumerate(vessels_csv_abs_path) if char=='/']
        if len(idxs)>0:
            file_prefix = vessels_csv_abs_path[idxs[-1]+1:]
        else:
            file_prefix = vessels_csv_abs_path
        file_prefix = file_prefix.replace("_vessel_array.csv", "")

        inp_data_dict = {
            'file_prefix': file_prefix,
            'vessels_csv_abs_path': vessels_csv_abs_path,
            'parameters_csv_abs_path': parameters_csv_abs_path,
            'model_type': 'cellml_only',
            'couple_to_1d': False,
            'external_modules_dir': None
        }

    else:
        if parse_inp_data_dict:
            inp_data_dict = yaml_parser.parse_user_inputs_file(inp_data_dict)

        file_prefix = inp_data_dict['file_prefix']
        # resources_dir = inp_data_dict['resources_dir']
        vessels_csv_abs_path = inp_data_dict['vessels_csv_abs_path']
        parameters_csv_abs_path = inp_data_dict['parameters_csv_abs_path']

    parser = CSV0DModelParser(inp_data_dict=inp_data_dict, parameter_id_dir=None)
    
    # print(file_prefix)
    # # print(resources_dir)
    # print(vessels_csv_abs_path)
    # print(parameters_csv_abs_path)
    # print(parser.vessel_filename)

    # csv_parser = CSVFileParser()
    # vessels_df = csv_parser.get_data_as_dataframe_multistrings(parser.vessel_filename, True)
    vessels_df = pd.read_csv(parser.vessel_filename, header=0, dtype=str) #, na_filter=False)
    vessels_df = vessels_df.fillna('')
    # print(vessels_df.head())
    print(vessels_df.keys())

    # contains_heart = vessels_df['name'].str.contains('heart', case=False, na=False).any()
    contains_heart = (vessels_df['name'].str.lower() == 'heart').any()
    if contains_heart==True:
        closeLoop = 1 # closed-loop CVS model
    else:
        closeLoop = 0
    print(f"closeLoop: {closeLoop}")

    # default model choices for CA modules selection
    heartModel = 'heart' # default heart model
    inflowType = 'nn_adan_2' # default inflow model
    
    termBC_art = 'pp' # default terminal BCs (arterial)
    termType_art = 'arterial_simple_novisco' # 'arterial_simple' # default terminal type (arterial)
    # in principle, we could use 'simple_novisco' for arterial terminals as well...
    termBC_ven = 'vp' # default terminal BCs (venous)
    termType_ven = 'simple_novisco' # default terminal BC type (venous)
    
    vessTL = 'nonlinear_visco' # default tube law model
    vessProp = 'material_prop_visco_const' # default vessel material properties model
    
    juncsModel = 'general' # default junction model
    # juncsModel='original' # otherwise

    if inp_data_dict==None:
        print("WARNING :: NO inp_data_dict passed as input to the function. Using default model choices then.")
    else:
        if closeLoop==1:
            if 'heartModel' in inp_data_dict.keys():
                heartModel = inp_data_dict['heartModel']
            else:
                print("WARNING :: heartModel NOT found in inp_data_dict, default value assigned.")
        else:
            if 'inflowType' in inp_data_dict.keys():
                inflowType = inp_data_dict['inflowType']
            else:
                print("WARNING :: inflowType NOT found in inp_data_dict, default value assigned.")
            
        # if 'termType' in inp_data_dict.keys():
        #     termType = inp_data_dict['termType']
        # else:
        #     print("WARNING :: termType NOT found in inp_data_dict, default value assigned.")
        
        if 'vessTubeLaw' in inp_data_dict.keys():
            vessTL = inp_data_dict['vessTubeLaw']
        else:
            print("WARNING :: vessTubeLaw NOT found in inp_data_dict, default value assigned.")
        
        if 'vessPropModel' in inp_data_dict.keys():
            vessProp = inp_data_dict['vessPropModel']
        else:
            print("WARNING :: vessPropModel NOT found in inp_data_dict, default value assigned.")
        
        if 'juncsModel' in inp_data_dict.keys():
            juncsModel = inp_data_dict['juncsModel']
        else:
            print("WARNING :: juncsModel NOT found in inp_data_dict, default value assigned.")
    
    if closeLoop==1:
        print(f"heartModel: {heartModel}")
    else:
        print(f"inflowType: {inflowType}")
    # print(f"termType: {termType}")
    print(f"termType (arteries): {termType_art} {termBC_art}")
    print(f"termType (veins): {termType_ven} {termBC_ven}")
    print(f"vessTL: {vessTL}")
    print(f"vessProp: {vessProp}")
    print(f"juncsModel: {juncsModel}")

    modules_default = [
        ['heart', 'vp', heartModel],
        ['par', 'vp', 'arterial_simple'],
        # ['par', 'vp', 'arterial_simple_novisco'],
        ['pvn', 'vp', 'venous'],
        # ['pvn', 'vp', 'venous_novisco']
    ]

    nVess = vessels_df.shape[0]

    vessels_df_new = vessels_df.copy()

    juncs = []
    juncs_in_out = []

    for i in range(nVess):

        vess_name = vessels_df_new.at[i,'name']
        inp_vess = vessels_df_new.at[i,'inp_vessels'].split()
        out_vess = vessels_df_new.at[i,'out_vessels'].split()
        # print(vess_name)
        # print(inp_vess)
        # print(out_vess)
        # print('\n')

        if (vess_name!='heart' and 'input_flow' not in vess_name): # and 'venous' not in vess_name):
            if inp_vess!=[]:
                if inp_vess[0]!='heart' and 'input_flow' not in inp_vess[0]: # and 'venous' not in inp_vess[0]:
                    junc_tmp = []
                    junc_in_out_tmp = []
                    
                    junc_tmp.append(vess_name)
                    junc_in_out_tmp.append(0)
                    
                    for j in range(len(inp_vess)):
                        junc_tmp.append(inp_vess[j])
                        junc_in_out_tmp.append(1)

                    for j in range(len(inp_vess)):
                        search_vess = inp_vess[j]
                        for k in range(nVess):
                            vess_name2 = vessels_df_new.at[k,'name']
                            inp_vess2 = vessels_df_new.at[k,'inp_vessels'].split()
                            out_vess2 = vessels_df_new.at[k,'out_vessels'].split()
                            if vess_name2 == search_vess: 
                                for h in range(len(junc_tmp)):
                                    if junc_tmp[h] in inp_vess2:
                                        for p in range(len(inp_vess2)):
                                            if inp_vess2[p] not in junc_tmp:
                                                junc_tmp.append(inp_vess2[p])
                                                junc_in_out_tmp.append(1)
                                        break
                                    elif junc_tmp[h] in out_vess2:
                                        for p in range(len(out_vess2)):
                                            if out_vess2[p] not in junc_tmp:
                                                junc_tmp.append(out_vess2[p])
                                                junc_in_out_tmp.append(0)
                                        break
                                break

                    matches = [junc for junc in juncs if Counter(junc) == Counter(junc_tmp)]
                    if matches:
                        pass
                    else:
                        juncs.append(junc_tmp)
                        juncs_in_out.append(junc_in_out_tmp)
            
            if out_vess!=[]:
                if out_vess[0]!='heart' and 'input_flow' not in out_vess[0]: # and 'venous' not in out_vess[0]:
                    junc_tmp = []
                    junc_in_out_tmp = []

                    junc_tmp.append(vess_name)
                    junc_in_out_tmp.append(1)

                    for j in range(len(out_vess)):
                        junc_tmp.append(out_vess[j])
                        junc_in_out_tmp.append(0)

                    for j in range(len(out_vess)):
                        search_vess = out_vess[j]
                        for k in range(nVess):
                            vess_name2 = vessels_df_new.at[k,'name']
                            inp_vess2 = vessels_df_new.at[k,'inp_vessels'].split()
                            out_vess2 = vessels_df_new.at[k,'out_vessels'].split()
                            if vess_name2 == search_vess: 
                                for h in range(len(junc_tmp)):
                                    if junc_tmp[h] in inp_vess2:
                                        for p in range(len(inp_vess2)):
                                            if inp_vess2[p] not in junc_tmp:
                                                junc_tmp.append(inp_vess2[p])
                                                junc_in_out_tmp.append(1)
                                        break
                                    elif junc_tmp[h] in out_vess2:
                                        for p in range(len(out_vess2)):
                                            if out_vess2[p] not in junc_tmp:
                                                junc_tmp.append(out_vess2[p])
                                                junc_in_out_tmp.append(0)
                                        break
                                break

                    matches = [junc for junc in juncs if Counter(junc) == Counter(junc_tmp)]
                    if matches:
                        pass
                    else:
                        juncs.append(junc_tmp)
                        juncs_in_out.append(junc_in_out_tmp)


    nJ = len(juncs)

    # for i in range(nJ):
    #     print(i, juncs[i])

    
    # Assign default BC_type and vessel_type to all vessels
    for i in range(nVess):
        
        mod_name = vessels_df_new.at[i,'name']
        
        if (closeLoop==1 and mod_name=='heart'):
            for j in range(len(modules_default)):
                if modules_default[j][0]=='heart':
                    vessels_df_new.at[i,'BC_type'] = modules_default[j][1]
                    vessels_df_new.at[i,'vessel_type'] = modules_default[j][2]
                    break
            # vessels_df_new.at[i,'BC_type'] = 'vp'
            # vessels_df_new.at[i,'vessel_type'] = heartModel
        
        elif (closeLoop==0 and 'input_flow' in mod_name):
            vessels_df_new.at[i,'BC_type'] = inflowType
            vessels_df_new.at[i,'vessel_type'] = 'inlet_flow'

        elif mod_name=='par':
            for j in range(len(modules_default)):
                if modules_default[j][0]=='par':
                    vessels_df_new.at[i,'BC_type'] = modules_default[j][1]
                    vessels_df_new.at[i,'vessel_type'] = modules_default[j][2]
                    break

        elif mod_name=='pvn':
            for j in range(len(modules_default)):
                if modules_default[j][0]=='pvn':
                    vessels_df_new.at[i,'BC_type'] = modules_default[j][1]
                    vessels_df_new.at[i,'vessel_type'] = modules_default[j][2]
                    break
        
        else:
            if closeLoop==1:
                if 'venous' in mod_name or mod_name.startswith('V_'): 
                    #XXX this needs to be imporved in the case we have an actual venous network as for its arterial counterpart
                    if mod_name.startswith('V_'): # assuming this name pattern is for veins
                        vessels_df_new.at[i,'BC_type'] = 'vp'
                        # vessels_df_new.at[i,'BC_type'] = 'vp_woCont'
                        vessels_df_new.at[i,'vessel_type'] = 'venous'
                        # vessels_df_new.at[i,'vessel_type'] = 'venous_novisco'
                    elif 'venous' in mod_name: # assuming this name pattern is for venous terminal compartments
                        vessels_df_new.at[i,'BC_type'] = termBC_ven+'_'+termType_ven
                        vessels_df_new.at[i,'vessel_type'] = 'Min_junction' # default for merging
                else:
                    if ('heart' in vessels_df_new.at[i,'inp_vessels'] 
                        or 'heart' in vessels_df_new.at[i,'out_vessels']):
                        if vessTL=='linear':
                            vessels_df_new.at[i,'BC_type'] = 'vv'
                        else:
                            vessels_df_new.at[i,'BC_type'] = 'vv_'+vessTL
                        vessels_df_new.at[i,'vessel_type'] = 'arterial'

                        if vessTL!='linear':
                            new_mod_name = 'K_tube_'+mod_name
                            new_row = {'name': new_mod_name,
                                'BC_type': 'nn',
                                'vessel_type': vessProp,
                                'inp_vessels': mod_name,
                                'out_vessels': ' '}
                            vessels_df_new = pd.concat([vessels_df_new, pd.DataFrame([new_row])], ignore_index=True)

                            out_vess = vessels_df_new.at[i,'out_vessels']
                            out_vess_new = out_vess+' '+new_mod_name
                            vessels_df_new.at[i,'out_vessels'] = out_vess_new

                    elif ('sink' in mod_name # assuming this name pattern is for arterial sinks
                            or mod_name.endswith('_T')): # assuming this name pattern is for arterial terminal compartments
                        if ('venous' in vessels_df_new.at[i,'inp_vessels'] 
                            or 'none' in vessels_df_new.at[i,'inp_vessels'] 
                            or vessels_df_new.at[i,'inp_vessels'].strip()==''):
                            vessels_df_new.at[i,'BC_type'] = termBC_art
                            vessels_df_new.at[i,'vessel_type'] = termType_art # 'terminal'
                        elif ('venous' in vessels_df_new.at[i,'out_vessels'] 
                            or 'none' in vessels_df_new.at[i,'out_vessels']
                            or vessels_df_new.at[i,'out_vessels'].strip()==''):
                            vessels_df_new.at[i,'BC_type'] = termBC_art
                            vessels_df_new.at[i,'vessel_type'] = termType_art # 'terminal'
                    
                    else:
                        if vessTL=='linear':
                            vessels_df_new.at[i,'BC_type'] = 'pv'
                        else:
                            vessels_df_new.at[i,'BC_type'] = 'pv_'+vessTL
                        vessels_df_new.at[i,'vessel_type'] = 'arterial'

                        if vessTL!='linear':
                            new_mod_name = 'K_tube_'+mod_name
                            new_row = {'name': new_mod_name,
                                'BC_type': 'nn',
                                'vessel_type': vessProp,
                                'inp_vessels': mod_name,
                                'out_vessels': ' '}
                            vessels_df_new = pd.concat([vessels_df_new, pd.DataFrame([new_row])], ignore_index=True)

                            out_vess = vessels_df_new.at[i,'out_vessels']
                            out_vess_new = out_vess+' '+new_mod_name
                            vessels_df_new.at[i,'out_vessels'] = out_vess_new
            else:
                if ('input_flow' in vessels_df_new.at[i,'inp_vessels'] or 'input_flow' in vessels_df_new.at[i,'out_vessels']):
                    if vessTL=='linear':
                        vessels_df_new.at[i,'BC_type'] = 'vv'
                    else:
                        vessels_df_new.at[i,'BC_type'] = 'vv_'+vessTL
                    vessels_df_new.at[i,'vessel_type'] = 'arterial'

                    if vessTL!='linear':
                        new_mod_name = 'K_tube_'+mod_name
                        new_row = {'name': new_mod_name,
                            'BC_type': 'nn',
                            'vessel_type': vessProp,
                            'inp_vessels': mod_name,
                            'out_vessels': ' '}
                        vessels_df_new = pd.concat([vessels_df_new, pd.DataFrame([new_row])], ignore_index=True)

                        out_vess = vessels_df_new.at[i,'out_vessels']
                        out_vess_new = out_vess+' '+new_mod_name
                        vessels_df_new.at[i,'out_vessels'] = out_vess_new

                elif vessels_df_new.at[i,'inp_vessels'].strip()=='':
                    vessels_df_new.at[i,'BC_type'] = termBC_art
                    vessels_df_new.at[i,'vessel_type'] = termType_art # 'terminal'
                elif vessels_df_new.at[i,'out_vessels'].strip()=='':
                    vessels_df_new.at[i,'BC_type'] = termBC_art
                    vessels_df_new.at[i,'vessel_type'] = termType_art # 'terminal'
                
                else:
                    if vessTL=='linear':
                        vessels_df_new.at[i,'BC_type'] = 'pv'
                    else:
                        vessels_df_new.at[i,'BC_type'] = 'pv_'+vessTL
                    vessels_df_new.at[i,'vessel_type'] = 'arterial'

                    if vessTL!='linear':
                        new_mod_name = 'K_tube_'+mod_name
                        new_row = {'name': new_mod_name,
                            'BC_type': 'nn',
                            'vessel_type': vessProp,
                            'inp_vessels': mod_name,
                            'out_vessels': ' '}
                        vessels_df_new = pd.concat([vessels_df_new, pd.DataFrame([new_row])], ignore_index=True)

                        out_vess = vessels_df_new.at[i,'out_vessels']
                        out_vess_new = out_vess+' '+new_mod_name
                        vessels_df_new.at[i,'out_vessels'] = out_vess_new

    
    # vessels_csv_abs_path = vessels_csv_abs_path.replace("_vessel_array.csv", "_MID_vessel_array.csv")
    # vessels_df_new.to_csv(vessels_csv_abs_path, index=False, header=True)
    
    nVessTot = vessels_df_new.shape[0]
    
    # Check for 0D junctions and, if necessary, change BC_type and vessel_type to vessels
    # converging to each junction to ensure we have admissible 0D junction configurations
    juncs_BCs = []
    juncs_idxs = []
    for k in range(nJ):
        junc = juncs[k]

        is_input_flow = -1
        is_heart = -1
        for kk in range(len(junc)):
            if 'input_flow' in junc[kk]:
                is_input_flow = 1
                break
            if 'heart' in junc[kk] and 'sink' not in junc[kk]:
                is_heart = 1
                break

        if (is_heart==-1 and is_input_flow==-1):
            nVJ = len(junc)
            
            junc_BCs = []
            junc_idxs = []
            for j in range(nVJ):
                nameVJ = junc[j]

                for i in range(nVess):
                    if vessels_df_new.at[i,'name']==nameVJ:
                        if juncs_in_out[k][j]==0:
                            junc_BCs.append(vessels_df_new.at[i,'BC_type'][:1])
                            junc_idxs.append(i)
                            break
                        elif juncs_in_out[k][j]==1:
                            junc_BCs.append(vessels_df_new.at[i,'BC_type'][1:2])
                            junc_idxs.append(i)
                            break
            juncs_BCs.append(junc_BCs)
            juncs_idxs.append(junc_idxs)
        else:
            juncs_BCs.append([])
            juncs_idxs.append([])


    # for k in range(nJ):
    #     if len(juncs[k])!=len(juncs_BCs[k]):
    #         print(juncs[k])
    #         print(juncs_in_out[k])
    #         print(juncs_BCs[k])
    #         print(juncs_idxs[k])
    #         print('\n')

    Min_junc_vess = []
    Nout_junc_vess = []
    for i in range(nJ):

        if len(juncs_idxs[i])>0:
            junc = juncs[i]
            nVJ = len(junc)
            
            if nVJ==2:
                if juncs_BCs[i][0]=='v' and juncs_BCs[i][1]=='p':
                    pass
                elif juncs_BCs[i][0]=='p' and juncs_BCs[i][1]=='v':
                    pass
                
                elif juncs_BCs[i][0]=='v' and juncs_BCs[i][1]=='v':
                    idx2 = juncs_idxs[i][1]
                    if juncs_in_out[i][1]==0:
                        bc_type = vessels_df_new.at[idx2,'BC_type']
                        bc_type_new = bc_type[:0]+'p'+bc_type[1:]
                        vessels_df_new.at[idx2,'BC_type'] = bc_type_new
                    elif juncs_in_out[i][1]==1:
                        bc_type = vessels_df_new.at[idx2,'BC_type']
                        bc_type_new = bc_type[:1]+'p'+bc_type[2:]
                        vessels_df_new.at[idx2,'BC_type'] = bc_type_new

                elif juncs_BCs[i][0]=='p' and juncs_BCs[i][1]=='p':
                    idx1 = juncs_idxs[i][0]
                    if juncs_in_out[i][0]==0:
                        bc_type = vessels_df_new.at[idx1,'BC_type']
                        bc_type_new = bc_type[:0]+'v'+bc_type[1:]
                        vessels_df_new.at[idx1,'BC_type'] = bc_type_new
                    elif juncs_in_out[i][0]==1:
                        bc_type = vessels_df_new.at[idx1,'BC_type']
                        bc_type_new = bc_type[:1]+'v'+bc_type[2:]
                        vessels_df_new.at[idx1,'BC_type'] = bc_type_new

            else:
                nJflow =  juncs_BCs[i].count('v')
         
                if nJflow==0:
                    idx0 = juncs_idxs[i][0]
                    if juncs_in_out[i][0]==0:
                        bc_type = vessels_df_new.at[idx0,'BC_type']
                        bc_type_new = bc_type[:0]+'v'+bc_type[1:]
                        vessels_df_new.at[idx0,'BC_type'] = bc_type_new
                        Min_junc_vess.append([idx0, nVJ])
                    elif juncs_in_out[i][0]==1:
                        bc_type = vessels_df_new.at[idx0,'BC_type']
                        bc_type_new = bc_type[:1]+'v'+bc_type[2:]
                        vessels_df_new.at[idx0,'BC_type'] = bc_type_new
                        Nout_junc_vess.append([idx0, nVJ])
                
                elif nJflow==1:
                    for j in range(nVJ):
                        if (juncs_in_out[i][j]==0 and juncs_BCs[i][j]=='v'):
                            Min_junc_vess.append([juncs_idxs[i][j],nVJ])
                            break
                        elif (juncs_in_out[i][j]==1 and juncs_BCs[i][j]=='v'):
                            Nout_junc_vess.append([juncs_idxs[i][j],nVJ])
                            break
                
                elif nJflow>1:
                    found = -1
                    for j in range(nVJ):
                        idxV = juncs_idxs[i][j]
                        if juncs_BCs[i][j]=='p':
                            pass
                        elif (juncs_in_out[i][j]==0 and juncs_BCs[i][j]=='v'):
                            if found==-1:
                                Min_junc_vess.append([idxV,nVJ])
                                found = 1
                            else:
                                bc_type = vessels_df_new.at[idxV,'BC_type']
                                bc_type_new = bc_type[:0]+'p'+bc_type[1:]
                                vessels_df_new.at[idxV,'BC_type'] = bc_type_new
                        elif (juncs_in_out[i][j]==1 and juncs_BCs[i][j]=='v'):
                            if found==-1:
                                Nout_junc_vess.append([idxV,nVJ])
                                found = 1
                            else:
                                bc_type = vessels_df_new.at[idxV,'BC_type']
                                bc_type_new = bc_type[:1]+'p'+bc_type[2:]
                                vessels_df_new.at[idxV,'BC_type'] = bc_type_new
                            
                
    print('Min-type junc vessels:', Min_junc_vess)
    print('Nout-type junc vessels:', Nout_junc_vess)

    MinNout_junc_vess = []
    if (len(Min_junc_vess)>0 and len(Nout_junc_vess)>0):
        for i in range(len(Min_junc_vess)):
            for j in range(len(Nout_junc_vess)):
                if Min_junc_vess[i][0]==Nout_junc_vess[j][0]:
                    print(Min_junc_vess[i], Nout_junc_vess[j])
                    MinNout_tmp = [Min_junc_vess[i][0], Min_junc_vess[i][1], Nout_junc_vess[j][1]]
                    if MinNout_tmp not in MinNout_junc_vess:
                        MinNout_junc_vess.append(MinNout_tmp)
                    break
                
    print('Min-Nout-type junc vessels:', MinNout_junc_vess)


    if len(MinNout_junc_vess)>0:
        for i in range(len(MinNout_junc_vess)):
            idxV = MinNout_junc_vess[i][0]
            nVJ1 = MinNout_junc_vess[i][1]
            nVJ2 = MinNout_junc_vess[i][2]
            if (nVJ1==3 and nVJ2==3):
                bc_type = vessels_df_new.at[idxV,'BC_type']
                bc_type_new = bc_type[:0]+'vv'+bc_type[2:]
                vessels_df_new.at[idxV,'BC_type'] = bc_type_new
                if juncsModel=='original':
                    vessels_df_new.at[idxV,'vessel_type'] = '2in2out_junction'
                elif juncsModel=='general':
                    vessels_df_new.at[idxV,'vessel_type'] = 'MinNout_junction'
            else:
                bc_type = vessels_df_new.at[idxV,'BC_type']
                bc_type_new = bc_type[:0]+'vv'+bc_type[2:]
                vessels_df_new.at[idxV,'BC_type'] = bc_type_new
                vessels_df_new.at[idxV,'vessel_type'] = 'MinNout_junction'
            
    if len(Min_junc_vess)>0:
        for i in range(len(Min_junc_vess)):
            idxV = Min_junc_vess[i][0]
            nVJ = Min_junc_vess[i][1]
            found = -1
            if len(MinNout_junc_vess)>0:
                for j in range(len(MinNout_junc_vess)):
                    Min_tmp = [MinNout_junc_vess[j][0], MinNout_junc_vess[j][1]]
                    if Min_junc_vess[i]==Min_tmp:
                        found = 1
                        break
            
            if found==-1:
                bc_type = vessels_df_new.at[idxV,'BC_type']
                bc_type_new = bc_type[:0]+'v'+bc_type[1:]
                vessels_df_new.at[idxV,'BC_type'] = bc_type_new
                if nVJ==3:
                    if juncsModel=='original':
                        vessels_df_new.at[idxV,'vessel_type'] = 'merge_junction'
                    elif juncsModel=='general':
                        vessels_df_new.at[idxV,'vessel_type'] = 'Min_junction'
                else:
                    vessels_df_new.at[idxV,'vessel_type'] = 'Min_junction'
        
    if len(Nout_junc_vess)>0:
        for i in range(len(Nout_junc_vess)):
            idxV = Nout_junc_vess[i][0]
            nVJ = Nout_junc_vess[i][1]
            found = -1
            if len(MinNout_junc_vess)>0:
                for j in range(len(MinNout_junc_vess)):
                    Nout_tmp = [MinNout_junc_vess[j][0], MinNout_junc_vess[j][2]]
                    if Nout_junc_vess[i]==Nout_tmp:
                        found = 1
                        break
            
            if found==-1:
                bc_type = vessels_df_new.at[idxV,'BC_type']
                bc_type_new = bc_type[:1]+'v'+bc_type[2:]
                vessels_df_new.at[idxV,'BC_type'] = bc_type_new
                if nVJ==3:
                    if juncsModel=='original':
                        vessels_df_new.at[idxV,'vessel_type'] = 'split_junction'
                    elif juncsModel=='general':
                        vessels_df_new.at[idxV,'vessel_type'] = 'Nout_junction'
                else:
                    vessels_df_new.at[idxV,'vessel_type'] = 'Nout_junction'


    vessels_csv_abs_path = vessels_csv_abs_path.replace("_vessel_array.csv", "_FINAL_vessel_array.csv")
    
    vessels_df_new.to_csv(vessels_csv_abs_path, index=False, header=True) # sep=',',

    print('DONE :: Vessel array file for model '+file_prefix+' filled and saved.')
    print("Exiting generate_vessel_array_fun() function.")


if __name__ == '__main__':
    try:
        generate_vessel_array_fun()

    except Exception as e:
        print(f"Failed to run python script generate_vessel_array.py: {e}", file=sys.stderr)
        exit()
