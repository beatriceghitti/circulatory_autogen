#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:38:24 2024

@author: bghi639
"""

import numpy as np
import json
import csv
import pandas as pd
import os
import sys
import copy
from configparser import ConfigParser


def generate1DPythonModelFiles(df_vess, df_params, vess_file, nodes_file, names_file, conn_1d_0d_info):

    # units conversion factors between 0D solver (Kg,m,s) and 1D solver (g,cm,s)
    convLen = 1.0e+2 # length units conversion from [m] to [cm]
    convQ = 1e+06 # flow units conversion from [m3/s] to [ml/s]=[cm3/s]
    convP = 1.0e+1 # pressure units conversion from [J/m3]=[Kg/m/s2] to [dyne/cm2]
    convR = convP/convQ # 1.0e-05 # resistance units conversion from [J s/m6] to [dyne s/cm5]
    convC = convQ/convP # 1.0e+05 # compliance units conversion from [m6/J] to [cm5/dyne]
    # convI = 1e-05 # inductance units conversion from [J s2/m6] to [dyne s2/cm5]
    
    # mmHgDyncm2 = 1333.2238 # pressure units conversion from [mmHg] to [dyne/cm2] for convenience
    # EeRef = 4.0e5 # [J/m3]

    nRow = df_vess.shape[0]
    nCol = df_vess.shape[1]

    print("1D vessel array file")
    print(type(df_vess))
    print('****')
    print(df_vess.dtypes)
    print('****')
    print(df_vess.head())
    print('****')
    print(df_vess.keys())
    print('****')
    print(nRow, nCol)
    print('***********************************************************************')

    nV = nRow
    print('Number of 1D vessels :', nV)
    print('****')

    vess = {}
    terms = []
    idxsV = []
    namesV = []

    for i in range(nV):
        nameV = df_vess.at[i,'name']
        idxsV.append(i)
        namesV.append(nameV)
        
        vess[i] = {'id' : i, 'name' : nameV,
                    'n1' : -1, 'n2' : -1,
                    'r_prox' : -1.0, 'r_dist' : -1.0, 'len' : -1.0, 
                    'Ee' : -1.0, 'Km' : -1.0, 'P0' : -1.0, 'Pext' : -1.0,
                    'type' : -1}
                
    # for i in range(nV):
    #     print(i, vess[i])
    # print('***********************************************************************')

    print('Writing vessel names file')
    df_names = pd.DataFrame(data={'vess_ID': idxsV, 'vess_name': namesV})
    df_names.to_csv(names_file, sep=',', index=False, header=True)  
    print('DONE!')
    print('****') 
    

    for i in range(nV):
        nameV = vess[i]['name']

        #XXX TODO improve this code below to differentiate between arterial and venous vessels, 
        # assuming that we do not have a 'vessel_type' in the CA vessel array file that can tell us this information
        if nameV.startswith(("A_","a_")) or any(sub in nameV for sub in ("art", "Art", "aort", "Aort")):
            vess[i]['type'] = 1 # artery
        elif nameV.startswith(("V_","v_")) or any(sub in nameV for sub in ("ven", "Ven", "vein", "Vein")):
            vess[i]['type'] = 0 # vein
        else:
            print(f"WARNING: art/ven type couldn't be determined for vessel {nameV}. Add it manually.")
            

        nameP = "l_"+nameV
        idxParam = df_params.index[df_params["variable_name"] == nameP].tolist()
        if len(idxParam)==1:
            vess[i]['len'] = float(df_params.at[idxParam[0],"value"])*convLen
        elif len(idxParam)==0:
            sys.exit(f"Parameter {nameP} not found in parameters dataframe.")
        elif len(idxParam)>1:
            sys.exit(f"Multiple matches for parameter {nameP} found in parameters dataframe.")

        nameP = "rProx_0_"+nameV
        idxParam = df_params.index[df_params["variable_name"] == nameP].tolist()
        nameP2 = "rDist_0_"+nameV
        idxParam2 = df_params.index[df_params["variable_name"] == nameP2].tolist()
        if len(idxParam)==1:
            if len(idxParam2)==1:
                vess[i]['r_prox'] = float(df_params.at[idxParam[0],"value"])*convLen
                vess[i]['r_dist'] = float(df_params.at[idxParam2[0],"value"])*convLen
            elif len(idxParam2)==0:
                sys.exit(f"Parameter {nameP} found in parameters dataframe, but not parameter {nameP2}. They should go in pair.")
            elif len(idxParam2)>1:
                sys.exit(f"Multiple matches for parameter {nameP2} found in parameters dataframe.")
        elif len(idxParam)>1:
            sys.exit(f"Multiple matches for parameter {nameP} found in parameters dataframe.")
        elif len(idxParam)==0:
            if len(idxParam2)==1:
                sys.exit(f"Parameter {nameP2} found in parameters dataframe, but not parameter {nameP}. They should go in pair.")
            elif len(idxParam2)>1:
                sys.exit(f"Parameter {nameP2} found in parameters dataframe (with multiple matches), but not parameter {nameP}. They should go in pair and be unique.")
            elif len(idxParam2)==0:
                nameP0 = "r_0_"+nameV
                print(f"Parameters {nameP} and {nameP2} not found in parameters dataframe. Searching for {nameP0}.")
                idxParam0 = df_params.index[df_params["variable_name"] == nameP0].tolist()
                if len(idxParam0)==1:
                    vess[i]['r_prox'] = float(df_params.at[idxParam0[0],"value"])*convLen
                    vess[i]['r_dist'] = float(df_params.at[idxParam0[0],"value"])*convLen
                elif len(idxParam0)==0:
                    sys.exit(f"Parameter {nameP0} not found in parameters dataframe.")
                elif len(idxParam0)>1:
                    sys.exit(f"Multiple matches for parameter {nameP0} found in parameters dataframe.")

        nameP = "E_K_tube_"+nameV
        idxParam = df_params.index[df_params["variable_name"] == nameP].tolist()
        nameP2 = "E_"+nameV
        idxParam2 = df_params.index[df_params["variable_name"] == nameP2].tolist()
        if len(idxParam)==1:
            vess[i]['Ee'] = float(df_params.at[idxParam[0],"value"])*convP
        elif len(idxParam)==0:
            if len(idxParam2)==1:
                print(f"Parameter {nameP} not found in parameters dataframe, parameter {nameP2} found instead.")
                vess[i]['Ee'] = float(df_params.at[idxParam2[0],"value"])*convP
            elif len(idxParam2)==0:
                # print(f"Parameters {nameP} and {nameP2} both not found in parameters dataframe. Assigning default value.")
                # vess[i]['Ee'] = EeRef*convP
                print(f"Parameters {nameP} and {nameP2} both not found in parameters dataframe. Assigning -1.0 by default.")
                vess[i]['Ee'] = -1.0 #XXX the 1D solver computes the stiffness parameter K from reference wave speed instead.
            elif len(idxParam2)>1:
                sys.exit(f"Parameter {nameP} not found in parameters dataframe. Parameter {nameP2} found instead but with multiple matches.")
        elif len(idxParam)>1:
            sys.exit(f"Multiple matches for parameter {nameP} found in parameters dataframe.")

        nameP = "K_m_K_tube_"+nameV
        idxParam = df_params.index[df_params["variable_name"] == nameP].tolist()
        nameP2 = "K_m_"+nameV
        idxParam2 = df_params.index[df_params["variable_name"] == nameP2].tolist()
        if len(idxParam)==1:
            vess[i]['Km'] = float(df_params.at[idxParam[0],"value"])*convP
        elif len(idxParam)==0:
            if len(idxParam2)==1:
                print(f"Parameter {nameP} not found in parameters dataframe, parameter {nameP2} found instead.")
                vess[i]['Km'] = float(df_params.at[idxParam2[0],"value"])*convP
            elif len(idxParam2)==0:
                print(f"Parameters {nameP} and {nameP2} both not found in parameters dataframe. Assigning 0.0 by default.")
                vess[i]['Km'] = 0.0
            elif len(idxParam2)>1:
                sys.exit(f"Parameter {nameP} not found in parameters dataframe. Parameter {nameP2} found instead but with multiple matches.")
        elif len(idxParam)>1:
            sys.exit(f"Multiple matches for parameter {nameP} found in parameters dataframe.")

        nameP = "u_0_"+nameV
        idxParam = df_params.index[df_params["variable_name"] == nameP].tolist()
        if len(idxParam)==1:
            vess[i]['P0'] = float(df_params.at[idxParam[0],"value"])*convP
        elif len(idxParam)==0:
            print(f"Parameter {nameP} not found in parameters dataframe. Using the reference pressure of input.ini (P0 for arteries, P0V for veins).")
            vess[i]['P0'] = np.nan #XXX the 1D solver then uses P0 (arteries) or P0V (veins) of input.ini
        elif len(idxParam)>1:
            sys.exit(f"Multiple matches for parameter {nameP} found in parameters dataframe.")

        nameP = "u_ext_"+nameV
        idxParam = df_params.index[df_params["variable_name"] == nameP].tolist()
        if len(idxParam)==1:
            vess[i]['Pext'] = float(df_params.at[idxParam[0],"value"])*convP
        elif len(idxParam)==0:
            print(f"Parameter {nameP} not found in parameters dataframe. Using the external pressure Pe of input.ini.")
            vess[i]['Pext'] = np.nan #XXX the 1D solver then uses Pe of input.ini
        elif len(idxParam)>1:
            sys.exit(f"Multiple matches for parameter {nameP} found in parameters dataframe.")


    # for i in range(nV):
    #     print(i, vess[i])
    # print('***********************************************************************')


    nodes = {}

    # Nodes with a 1d-0d connection (code 999): one node per connection, with all the 1D vessel ends at that node.
    # They are built from whichever of these vessels comes first in the vessel array, so they do not depend on the order
    # of the vessels (e.g. a 1D vessel splitting into a 1D vessel and a 0D module, or a 1D vessel and a 0D module
    # merging into a 1D vessel, where the connection belongs to only one of the two 1D vessels).
    idx_by_name = {vess[k]['name'] : k for k in range(nV)} # 1D vessels only (0D modules and BCs are not in it)
    conn_ends = [] # 1D vessel ends with a 1d-0d connection: (vessel index, 0 inlet / 1 outlet)
    for j in range(len(conn_1d_0d_info)):
        conn_ends.append((conn_1d_0d_info[str(j+1)]['vess1d_idx'], conn_1d_0d_info[str(j+1)]['vess1d_bc_in0_or_out1']))

    def node_ends(iV, side): # OK
        """
        Finds every 1D vessel end at the same node as a given end
        All the 1D vessel ends at the node of the end (iV, side) of vessel iV (side 0: inlet, 1: outlet), (iV, side) first.
        Two ends are at the same node when their vessels list each other, e.g. the outlet of A and the inlet of B
        when A has B in out_vessels and B has A in inp_vessels.
        The search follows these pairs, so that it finds the same ends whichever one it starts from.
        """
        ends = [(iV, side)]
        k = 0
        while k < len(ends):
            iA, sideA = ends[k]
            nameA = vess[iA]['name']
            listA = df_vess.at[iA,'inp_vessels'] if sideA == 0 else df_vess.at[iA,'out_vessels']
            for nameB in listA:
                if nameB not in idx_by_name:
                    continue # 0D module or BC
                iB = idx_by_name[nameB]
                # the end of B that lists A: usually its outlet for an inlet of A, and its inlet for an outlet of A
                if sideA == 0:
                    sidesB = ((1, df_vess.at[iB,'out_vessels']), (0, df_vess.at[iB,'inp_vessels']))
                else:
                    sidesB = ((0, df_vess.at[iB,'inp_vessels']), (1, df_vess.at[iB,'out_vessels']))
                for sideB, listB in sidesB:
                    if nameA in listB:
                        if (iB, sideB) not in ends:
                            ends.append((iB, sideB))
                        break
            k += 1
        return ends

    def add_hybrid_node(iN, iV, side): # OK
        """
        Creates node iN as a node with a 1d-0d connection (code 999) if the end (iV, side) of vessel iV, or another 1D
        vessel end at its node, has a 1d-0d connection, and puts all the 1D vessel ends at that node on it.
        Returns False, without creating anything, if none of them has a 1d-0d connection.
        """
        ends = node_ends(iV, side)
        conns_at_node = [e for e in conn_ends if e in ends] # one entry per connection (one end may have several)
        if len(conns_at_node) == 0:
            # NO 1D vessel end at this node has a 1d-0d connection.
            return False
        if len(conns_at_node) > 1:
            sys.exit(f"generate1DPythonModelFiles :: ERROR: {len(conns_at_node)} 1d-0d connections at the same node, on "
                     f"{', '.join(vess[e[0]]['name'] + (' inlet' if e[1] == 0 else ' outlet') for e in conns_at_node)}: "
                     f"only one 1d-0d connection per node is possible")
        nodes[iN] = {'id' : iN, 'type' : 999, 'vess_list' : []} # code 999: hybrid junction / 1d-0d connection
        for (iB, sideB) in ends:
            vess[iB]['n1' if sideB == 0 else 'n2'] = iN
            nodes[iN]['vess_list'].append(iB)
            if iB != iV:
                print(f"Another 1D vessel found in hybrid connection: {vess[iB]['name']}")
        return True

    # boundary conditions in the 1D vessel array, and the end of a vessel where each one goes (0: inlet, in inp_vessels;
    # 1: outlet, in out_vessels)
    BC_sides = {'input_flow_BC' : 0, 'input_pressure_BC' : 0, 'output_flow_BC' : 1, 'output_pressure_BC' : 1}

    def end_str(iA, sideA):
        return f"the {'inlet' if sideA == 0 else 'outlet'} of 1D vessel {vess[iA]['name']}"

    def check_1d_junction(iV, side): # OK
        """
        The node of the end (iV, side) has no 1d-0d connection and is not a boundary condition, so it must be a 1D junction:
        every 1D vessel end at that node must list only 1D vessels (in inp_vessels for an inlet, in out_vessels for an outlet).
        Stops with an error otherwise, instead of writing a wrong network.
        """
        for (iA, sideA) in node_ends(iV, side):
            col = 'inp_vessels' if sideA == 0 else 'out_vessels'
            listA = df_vess.at[iA,col]
            if len(listA) == 0:
                sys.exit(f"generate1DPythonModelFiles :: ERROR: {end_str(iA, sideA)} has no vessel or boundary condition in {col}")
            for nameB in listA:
                if nameB in BC_sides:
                    if BC_sides[nameB] != sideA:
                        sys.exit(f"generate1DPythonModelFiles :: ERROR: {end_str(iA, sideA)} has the boundary condition {nameB} in {col}: "
                                 f"input_*_BC boundary conditions go in inp_vessels, output_*_BC ones in out_vessels")
                    else:
                        sys.exit(f"generate1DPythonModelFiles :: ERROR: {end_str(iA, sideA)} has the boundary condition {nameB} "
                                 f"together with other vessels in {col} ({' '.join(listA)}): a boundary condition must be alone")
                elif nameB not in idx_by_name:
                    sys.exit(f"generate1DPythonModelFiles :: ERROR: {end_str(iA, sideA)} has {nameB} in {col}, which is not a 1D vessel "
                             f"or a boundary condition, but this end has no 1d-0d connection (check that the 1D vessel array and the "
                             f"1d-0d connections come from the same model)")

    # every 1D vessel listed by another 1D vessel (in inp_vessels or out_vessels) must list it back: two 1D vessels share
    # a node only when they list each other (see node_ends())
    for k in range(nV):
        for col in ('inp_vessels', 'out_vessels'):
            for nameB in df_vess.at[k,col]:
                if nameB in idx_by_name:
                    iB = idx_by_name[nameB]
                    if vess[k]['name'] not in df_vess.at[iB,'inp_vessels'] and vess[k]['name'] not in df_vess.at[iB,'out_vessels']:
                        sys.exit(f"generate1DPythonModelFiles :: ERROR: 1D vessel {vess[k]['name']} has {nameB} in {col}, but 1D vessel "
                                 f"{nameB} does not have {vess[k]['name']} in its inp_vessels or out_vessels")

    iN = 0
    for i in range(nV):
        iV = vess[i]['id']
        nameV = vess[i]['name']
        n1 = vess[i]['n1']
        n2 = vess[i]['n2']

        # print(iV, nameV)
        
        if n1==-1:
            vess[i]['n1'] = iN

            inp_vess = df_vess.at[i,'inp_vessels']
            out_vess = df_vess.at[i,'out_vessels']

            # node with a 1d-0d connection (code 999) if this inlet, or another 1D vessel end at its node, has one
            if add_hybrid_node(iN, iV, 0):
                iN +=1

            else:
                # code 0: inflow BC
                # code 1: inlet pressure BC
                if (len(inp_vess)==1 and inp_vess[0]=='input_flow_BC'):
                    nodes[iN] = {'id' : iN, 'type' : 0, 'vess_list' : [iV]} # code 0: inflow BC
                    iN +=1
                elif (len(inp_vess)==1 and inp_vess[0]=='input_pressure_BC'):
                    nodes[iN] = {'id' : iN, 'type' : 1, 'vess_list' : [iV]} # code 1: inlet pressure BC
                    iN +=1

                # elif (len(inp_vess)==1 and inp_vess[0]=='output_flow_BC'):
                #     nodes[iN] = {'id' : iN, 'type' : 0, 'vess_list' : [iV]}
                # elif (len(inp_vess)==1 and inp_vess[0]=='output_pressure_BC'):
                #     nodes[iN] = {'id' : iN, 'type' : 1, 'vess_list' : [iV]}
                
                else:
                    # 1D junction: stops with an error if the vessel ends at this node do not form one (see check_1d_junction())
                    check_1d_junction(iV, 0)

                    idVJ_list = [iV]
                    nameVJ_list = [nameV]
                    sideVJ_list = [-1]
                    for j in range(len(inp_vess)):
                        nameV2 = inp_vess[j]
                        iV2 = [kk for kk in range(nV) if vess[kk]['name']==nameV2][0] # a 1D vessel (see check_1d_junction())
                        nameVJ_list.append(nameV2)
                        idVJ_list.append(iV2)

                    for j in range(len(inp_vess)):
                        search_vess = inp_vess[j]
                        for k in range(nV):
                            nameV2 = vess[k]['name']
                            inp_vess2 = df_vess.at[k,'inp_vessels']
                            out_vess2 = df_vess.at[k,'out_vessels']
                            if nameV2 == search_vess: 
                                for h in range(len(nameVJ_list)):
                                    if (nameVJ_list[h]!=nameV2 and nameVJ_list[h] in inp_vess2):
                                        vess[k]['n1'] = iN

                                        for p in range(len(inp_vess2)):
                                            nameV3 = inp_vess2[p]
                                            iV3 = [kk for kk in range(nV) if vess[kk]['name']==nameV3][0]
                                            if nameV3 not in nameVJ_list:
                                                inp_vess3 = df_vess.at[iV3,'inp_vessels']
                                                out_vess3 = df_vess.at[iV3,'out_vessels']
                                                if any(x in nameVJ_list for x in inp_vess3):
                                                    vess[iV3]['n1'] = iN
                                                elif any(x in nameVJ_list for x in out_vess3):
                                                    vess[iV3]['n2'] = iN
                                                nameVJ_list.append(nameV3)
                                                idVJ_list.append(iV3)  
                                        
                                        sideVJ_list.append(-1)
                                        break
                                    
                                    elif (nameVJ_list[h]!=nameV2 and nameVJ_list[h] in out_vess2):
                                        vess[k]['n2'] = iN
                                        
                                        for p in range(len(out_vess2)):
                                            nameV3 = out_vess2[p]
                                            iV3 = [kk for kk in range(nV) if vess[kk]['name']==nameV3][0]
                                            if nameV3 not in nameVJ_list:
                                                inp_vess3 = df_vess.at[iV3,'inp_vessels']
                                                out_vess3 = df_vess.at[iV3,'out_vessels']
                                                if any(x in nameVJ_list for x in inp_vess3):
                                                    vess[iV3]['n1'] = iN
                                                elif any(x in nameVJ_list for x in out_vess3):
                                                    vess[iV3]['n2'] = iN
                                                nameVJ_list.append(nameV3)
                                                idVJ_list.append(iV3)
                                        
                                        sideVJ_list.append(1)
                                        break
                                break
                    
                    nVJ = len(idVJ_list)
                    idVJ_list.sort()
                    nodes[iN] = {'id' : iN, 'type' : nVJ, 'vess_list' : idVJ_list} # code N>1: 1D junction with N vessels
                    if nVJ==2:
                        if sideVJ_list[0]==sideVJ_list[1]:
                            print(idVJ_list,sideVJ_list)
                    elif nVJ==3:
                        if sideVJ_list[0]==sideVJ_list[1] and sideVJ_list[0]==sideVJ_list[2]:
                            print(idVJ_list,sideVJ_list)
                    # print("fully 1d junction node:",iN,nVJ,idVJ_list,sideVJ_list)
                    iN +=1
            
        if n2==-1:
            vess[i]['n2'] = iN

            inp_vess = df_vess.at[i,'inp_vessels']
            out_vess = df_vess.at[i,'out_vessels']

            # node with a 1d-0d connection (code 999) if this outlet, or another 1D vessel end at its node, has one
            if add_hybrid_node(iN, iV, 1):
                iN +=1

            else:
                # code -1: RCR / single-resistance / outlet pressure BC
                # code -2: outflow BC
                if (len(out_vess)==1 and out_vess[0]=='output_flow_BC'):
                    nodes[iN] = {'id' : iN, 'type' : -2, 'vess_list' : [iV]} # code -2: outflow BC
                    iN +=1
                elif (len(out_vess)==1 and out_vess[0]=='output_pressure_BC'):
                    nodes[iN] = {'id' : iN, 'type' : -1, 'vess_list' : [iV]} # code -1: RCR / single-resistance / outlet pressure BC
                    iN +=1

                # elif (len(out_vess)==1 and out_vess[0]=='input_flow_BC'):
                #     nodes[iN] = {'id' : iN, 'type' : 0, 'vess_list' : [iV]}
                # elif (len(out_vess)==1 and out_vess[0]=='input_pressure_BC'):
                #     nodes[iN] = {'id' : iN, 'type' : 1, 'vess_list' : [iV]}

                else:
                    # 1D junction: stops with an error if the vessel ends at this node do not form one (see check_1d_junction())
                    check_1d_junction(iV, 1)

                    idVJ_list = [iV]
                    nameVJ_list = [nameV]
                    sideVJ_list = [1]
                    for j in range(len(out_vess)):
                        nameV2 = out_vess[j]
                        iV2 = [kk for kk in range(nV) if vess[kk]['name']==nameV2][0] # a 1D vessel (see check_1d_junction())
                        nameVJ_list.append(nameV2)
                        idVJ_list.append(iV2)

                    for j in range(len(out_vess)):
                        search_vess = out_vess[j]
                        for k in range(nV):
                            nameV2 = vess[k]['name']
                            inp_vess2 = df_vess.at[k,'inp_vessels']
                            out_vess2 = df_vess.at[k,'out_vessels']
                            if nameV2 == search_vess: 
                                for h in range(len(nameVJ_list)):
                                    if (nameVJ_list[h]!=nameV2 and nameVJ_list[h] in inp_vess2):
                                        vess[k]['n1'] = iN
                                        
                                        for p in range(len(inp_vess2)):
                                            nameV3 = inp_vess2[p]
                                            iV3 = [kk for kk in range(nV) if vess[kk]['name']==nameV3][0]
                                            if nameV3 not in nameVJ_list:
                                                inp_vess3 = df_vess.at[iV3,'inp_vessels']
                                                out_vess3 = df_vess.at[iV3,'out_vessels']
                                                if any(x in nameVJ_list for x in inp_vess3):
                                                    vess[iV3]['n1'] = iN
                                                elif any(x in nameVJ_list for x in out_vess3):
                                                    vess[iV3]['n2'] = iN
                                                nameVJ_list.append(nameV3)
                                                idVJ_list.append(iV3)
                                        
                                        sideVJ_list.append(-1)
                                        break
                                    
                                    elif (nameVJ_list[h]!=nameV2 and nameVJ_list[h] in out_vess2):
                                        vess[k]['n2'] = iN

                                        for p in range(len(out_vess2)):
                                            nameV3 = out_vess2[p]
                                            iV3 = [kk for kk in range(nV) if vess[kk]['name']==nameV3][0]
                                            if nameV3 not in nameVJ_list:
                                                inp_vess3 = df_vess.at[iV3,'inp_vessels']
                                                out_vess3 = df_vess.at[iV3,'out_vessels']
                                                if any(x in nameVJ_list for x in inp_vess3):
                                                    vess[iV3]['n1'] = iN
                                                elif any(x in nameVJ_list for x in out_vess3):
                                                    vess[iV3]['n2'] = iN
                                                nameVJ_list.append(nameV3)
                                                idVJ_list.append(iV3)
                                        
                                        sideVJ_list.append(1)
                                        break
                                break
                    
                    nVJ = len(idVJ_list)
                    idVJ_list.sort()
                    nodes[iN] = {'id' : iN, 'type' : nVJ, 'vess_list' : idVJ_list} # code N>1: 1D junction with N vessels
                    if nVJ==2:
                        if sideVJ_list[0]==sideVJ_list[1]:
                            print(idVJ_list,sideVJ_list)
                    elif nVJ==3:
                        if sideVJ_list[0]==sideVJ_list[1] and sideVJ_list[0]==sideVJ_list[2]:
                            print(idVJ_list,sideVJ_list)
                    # print("fully 1d junction node:",iN,nVJ,idVJ_list,sideVJ_list)
                    iN +=1

            
    nN = len(nodes)
    print('Number of nodes :', nN)
    print('****')

    print("1D vessels:")
    for iV in range(nV):
        print(iV, vess[iV])
    print('***********************************************************************') 
    print("1D network nodes:")
    for iN in range(nN):
        print(iN, nodes[iN])
    print('***********************************************************************')


    print('Writing vessels file')
    f = open(vess_file, "w")

    # f.write("# 0: vess ID; 1: first node; 2: second node; 3: length [cm]; 4: inlet radius [cm]; 5: outlet radius [cm]; 6: tot term resistance [dyn s/cm5]; 7: term compliance [cm5/dyn]\n")
    # f.write("# 0: vess_ID; 1: first_node; 2: second_node; 3: length [cm]; 4: inlet_radius [cm]; 5: outlet_radius [cm]; 6: tot_term_resistance [dyn s/cm5]; 7: term_compliance [cm5/dyn]; 8: wall_thickness [cm]; 9: elastic_mod [dyn/cm2]\n")
    # vessel file of the Python 1D solver (run1DBFmodel.py), with nColV = 14 columns (nColV in input.ini): the reference
    # and external pressures of each vessel (NaN: those of input.ini), and the arterial (1) / venous (0) type in the last column
    f.write("# 0: vess_ID; 1: first_node; 2: second_node; "
            "3: length [cm]; 4: inlet_radius [cm]; 5: outlet_radius [cm]; "
            "6: tot_term_resistance [dyn s/cm5]; 7: term_compliance [cm5/dyn]; "
            "8: wall_thickness [cm]; 9: elastic_mod [dyn/cm2]; 10: tilt_angle [deg]; "
            "11: p_0 [dyn/cm2]; 12: p_ext [dyn/cm2]; 13: art_ven_type "
            "\n"
        )


    for i in range(nV):
        iV = int(vess[i]['id'])
        nameV = vess[i]['name']
        
        f.write("%.i " % (vess[i]['id']))
        f.write("%.i " % (vess[i]['n1']))
        f.write("%.i " % (vess[i]['n2']))
        
        f.write("%.18e " % (vess[i]['len']))
        
        f.write("%.18e " % (vess[i]['r_prox']))
        f.write("%.18e " % (vess[i]['r_dist']))
        
        f.write("%.18e " % (-1.0)) # total terminal resistance
        f.write("%.18e " % (-1.0)) # terminal compliance
        
        f.write("%.18e " % (-1.0)) # wall thickness - not provided if directly computed by the 1D BFM solver

        f.write("%.18e " % (vess[i]['Ee'])) # elastic/Young modulus (used with Ee/EeV < 0 in input.ini)
        f.write("%.18e " % (0.0)) # tilt angle relative to the horizontal (used only with gravity)

        f.write("%.18e " % (vess[i]['P0'])) # reference pressure (NaN: P0/P0V of input.ini)
        f.write("%.18e " % (vess[i]['Pext'])) # external pressure (NaN: Pe of input.ini)

        f.write("%.i " % (vess[i]['type']))

        f.write("\n")

    f.close()
    print('DONE!')
    print('****') 


    print('Writing nodes file')
    f = open(nodes_file, "w")

    f.write("# 0: node_ID; 1: node_type; 2: x_coord [cm]; 3: y_coord [cm]; 4: z_coord [cm] \n")

    for i in range(nN):
        f.write("%.i " % (nodes[i]['id']))
        f.write("%.i " % (nodes[i]['type']))
        f.write("%.18e " % (0.0))
        f.write("%.18e " % (0.0))
        f.write("%.18e " % (0.0))
        f.write("\n")

    f.close()
    print('DONE!')
    print('****')

    return vess, nodes


def generate1DPythonSimInitFile(df_params, vess, nodes, sim_init_file, filename_prefix, run_folder, ODEsolver, dtSample, computeTotBV,
                                T0, pre_time, sim_time, petsc_solver='CN', inflow_func_1d=None, bc_modules_1d=None):
    # T0: cardiac period, parameter T of the model (get_cardiac_period() in CVSCppGenerator.py)
    # pre_time, sim_time: time window of the user inputs (get_time_window() in CVSCppGenerator.py)
    # petsc_solver: PETSc time integration scheme with ODEsolver PETSC, the same as in the 0d solver
    #               (get_petsc_solver() in CVSCppGenerator.py)
    # inflow_func_1d: built-in inflow function of the 1D solver (inflowFunc in input.ini) for the inflow BCs (inlet_flow
    #                 module at the inlet of a 1D vessel), from the user inputs; required if the 1D network has a
    #                 time-dependent one (a constant one, BC_type nn_constant, is a constant inflow Qin instead)
    # bc_modules_1d: boundary modules at the 1D vessels, (1D vessel name, 'inlet'/'outlet') -> module (name, vessel_type,
    #                BC_type), from the model parser (ModelParsers.split_0d_1d_vessel_array()); the values of the
    #                constant inflow, inlet pressure, outlet pressure and outflow BCs are their parameters

    config = ConfigParser(allow_no_value=True, delimiters=(':',)) # delimiters=('=', ':')
    config.optionxform = str  # keep keys case-sensitive
    
    # print(filename_prefix)
    # print(run_folder)
    
    # input.ini of the Python 1D solver (run1DBFmodel.py)
    params = 'constant'
    # params = 'linear'
    nVar1D = 5 # area, flow, wall stiffness, reference area, external pressure (6 with gravity)

    # 'wallthickness' = 1, if vessel wall thickness to be computed; 'wallthickness' = 0, if provided in the vess file
    wallthickness = 1
    
    config['network'] = {
        'networkName': filename_prefix[:-3],
        'testFold': run_folder.rstrip('/')+'/', # run folder (with '/'), with the input files folder and the results folder
        'inputFold': 'NA', # input files in testFold/inputFoldName/
        'inputFoldName': 'input_files',
        'vessFile': 'vess_'+filename_prefix[:-3]+'.txt',
        'nodeFile': 'nodes_'+filename_prefix[:-3]+'.txt',
        'nameFile': 'names_'+filename_prefix[:-3]+'.csv',
        'inflowFile_prefix': 'NA',
        'outflowFile_prefix': 'NA',
        'nVess': len(vess),
        'nNode': len(nodes),
        'nVar1D': nVar1D,
        'nVarJ': 2,
        'nColV': 14, # columns of the vessel file (see generate1DPythonModelFiles())
        'params': params,
        'wallthickness': wallthickness,
        'gravity': 0, # gravity is not supported in a coupled 1D-0D simulation
        'couple_to_0d': True,
        'compute_tot_volume': computeTotBV
    }

    tEnd = pre_time + sim_time
    
    config['discretization'] = {
        'nCellsMin': 2,
        'nCells': -1,
        'dxMax': 1.0, #0.1,
        'NMAX': 10000000000000,
        'T0': T0,
        'tIni': 0.0,
        'tEnd': tEnd,
        'dtSample': dtSample, 
        'tolTime': 1e-11, # 1e-10
        'checkPeriod': 0 # the 1D model runs from tIni to tEnd (in a coupled run, from 0 to pre_time + sim_time)
    }
    # In a coupled run, the 1d solver gets T0, pre_time and sim_time from the coupler instead (coupler_config.json,
    # written with the same values), and checks that T0 is the same as here.
    # tolTime is used only when running the 1d solver alone: in a coupled run, the 1d solver
    # uses the time tolerance of the coupler (COUPLING_TOL_TIME in src/coupler/coupler_mpi_common.h).

    mmHgDyncm2 = 1333.2238 # pressure units conversion from [mmHg] to [dyne/cm2] for convenience

    rho = 1.040
    for i in range(df_params.shape[0]):
        if (df_params.at[i,'variable_name']=='rho' and df_params.at[i,'units']=='Js2_per_m5'):
            rho = float(df_params.at[i,'value'])*1e-3
            break
    mu = 0.040
    for i in range(df_params.shape[0]):
        if (df_params.at[i,'variable_name']=='mu' and df_params.at[i,'units']=='Js_per_m3'):
            mu = float(df_params.at[i,'value'])*1e+1
            break
    nu = 0.5
    for i in range(df_params.shape[0]):
        if (df_params.at[i,'variable_name']=='nu' and df_params.at[i,'units']=='dimensionless'):
            nu = float(df_params.at[i,'value'])
            break
    EeRefA = 4.0e6 # [dyn/cm2] # 2.25e6 [dyn/cm2]
    for i in range(len(vess)):
        if 'Ee' in vess[i] and vess[i]['type']==1: # arterial vessel
            if vess[i]['Ee']>=0.:
                EeRefA = float(vess[i]['Ee'])
                break
    EeRefV = 1.0e5 # [dyn/cm2]
    for i in range(len(vess)):
        if 'Ee' in vess[i] and vess[i]['type']==0: # venous vessel
            if vess[i]['Ee']>=0.:
                EeRefV = float(vess[i]['Ee'])
                break
    P0A = 1.0e+5 # [dyn/cm2] ~75 [mmHg]
    for i in range(len(vess)):
        if 'P0' in vess[i] and vess[i]['type']==1: # arterial vessel
            if vess[i]['P0']>=0.:
                P0A = float(vess[i]['P0'])
                break
    P0V = 6666. # [dyn/cm2] ~5 [mmHg]
    for i in range(len(vess)):
        if 'P0' in vess[i] and vess[i]['type']==0: # venous vessel
            if vess[i]['P0']>=0.:
                P0V = float(vess[i]['P0'])
                break

    # Ee, EeV < 0: Young's modulus of each vessel from the vessel file (elastic_mod);
    # P0, P0V, Pe: reference and external pressures of the vessels without their own values in the vessel file (p_0, p_ext = NaN)
    config['model'] = {
        'mmHgDyncm2': mmHgDyncm2,
        'rho': rho,
        'mu': mu,
        'nu': nu,
        'velProf': 2.0,
        'coriolis': 1.0,
        'tubeLaw': 'alastruey', # arteries (veins always use the alastruey tube law)
        'Ee': -1.0,
        'm': 0.5,
        'n': 0.0,
        'EeV': -1.0,
        'mV': 10.0,
        'nV': -1.5,
        'pries': 0,
        'couplePtot': 0, # 1,
        'P0': P0A,
        'P0V': P0V,
        'PiniA': P0A,
        'PiniV': P0V,
        'Pe': 0.0,
        'Pv': 6666.0
    }


    config['ICs'] = {
        'ICtype': 1,
        'ICstate': -1
    }
    # 'ICtype' = 0 if Pinit=0; 1 if Pinit=PiniA/PiniV; 2 if Pinit=P0A/P0V


    nIn = 0
    idxs_nIn = []
    nOut = 0
    idxs_nOut = []
    for i in range(len(nodes)):
        if (nodes[i]['type']==0 or nodes[i]['type']==1):
            nIn +=1
            idxs_nIn.append(nodes[i]['id'])
        elif (nodes[i]['type']==-1 or nodes[i]['type']==-3 or nodes[i]['type']==-4):
            nOut +=1
            idxs_nOut.append(nodes[i]['id'])

    idxs_vIn = []
    if nIn>0:
        for i in range(nIn):
            nID = idxs_nIn[i]
            for j in range(len(vess)):
                if (vess[j]['n1']==nID or vess[j]['n2']==nID):
                    idxs_vIn.append( str(vess[j]['id']) )
                    break

    idxs_vOut = []
    if nOut>0:
        for i in range(nOut):
            nID = idxs_nOut[i]
            for j in range(len(vess)):
                if (vess[j]['n1']==nID or vess[j]['n2']==nID):
                    idxs_vOut.append( str(vess[j]['id']) )
                    break

    # inflow BCs (code 0: inlet_flow module at the inlet of a 1D vessel). The 1D solver takes one inflow type for all its
    # inflow BCs (inflowFunc in [BCs]):
    # - constant inlet_flow modules (BC_type nn_constant): constant inflow (inflowFunc constant), with the constant value
    #   of the modules, i.e. their parameter v_<module>, as Qin (see the values of the BCs below);
    # - time-dependent inlet_flow modules (e.g. nn_fourier_8, nn_aorticbif): one of the built-in inflow functions of the
    #   1D solver, inflow_func_1d of the user inputs, with the cardiac period T0 (the 1D solver checks the name at
    #   start-up). If there are also constant ones, they use the same function.
    #XXX The parameters of the time-dependent inlet_flow modules are not used, as the equivalent inlet flow function should be available in the 1D solver.
    vessInflow = [vess[j]['name'] for i in range(len(nodes)) if nodes[i]['type']==0
                  for j in range(len(vess)) if vess[j]['n1']==nodes[i]['id']]
    modulesInflow = []
    for nameV in vessInflow:
        module = (bc_modules_1d or {}).get((nameV, 'inlet'))
        if module is None:
            sys.exit(f"generate1DPythonSimInitFile :: ERROR: no boundary module found at the inlet of the 1D vessel "
                     f"{nameV}, which has an inflow BC (code 0)")
        modulesInflow.append(module)
    inflowConst = [module['name'] for module in modulesInflow if module['BC_type']=='nn_constant']
    inflowTime = [f"{module['name']} (BC_type {module['BC_type']})" for module in modulesInflow
                  if module['BC_type']!='nn_constant']
    if inflowTime:
        if not inflow_func_1d:
            sys.exit(f"generate1DPythonSimInitFile :: ERROR: the 1D network has a time-dependent inflow BC (inlet_flow "
                     f"module(s) {', '.join(inflowTime)}) at the inlet of the 1D vessel(s) {', '.join(vessInflow)}: give "
                     "inflow_func_1d in the user inputs yaml file, the name of a built-in inflow function of the 1D "
                     "solver (e.g. inflow_aorticbif, see BUILTIN_INFLOW_FUNCS in vessel.py of the 1D solver)")
        inflowFunc = inflow_func_1d
        print(f"generate1DPythonSimInitFile :: NOTE: inflow BC at the inlet of the 1D vessel(s) {', '.join(vessInflow)}: "
              f"the 1D solver uses its built-in inflow function {inflow_func_1d} (inflow_func_1d of the user inputs), with "
              f"period T = {T0}; the parameters of the inlet_flow module(s) are not used")
        if inflowConst:
            print(f"generate1DPythonSimInitFile :: WARNING: the 1D solver takes one inflow type for all its inflow BCs, "
                  f"but the 1D network has both time-dependent (inlet_flow module(s) {', '.join(inflowTime)}) and constant "
                  f"(inlet_flow module(s) {', '.join(inflowConst)}) inflow BCs: the constant ones also use the built-in "
                  f"inflow function {inflow_func_1d}")
    elif inflowConst:
        inflowFunc = 'constant'
        if inflow_func_1d:
            print(f"generate1DPythonSimInitFile :: WARNING: inflow_func_1d = {inflow_func_1d} of the user inputs is not "
                  f"used: the inflow BCs of the 1D network are constant (inlet_flow module(s) {', '.join(inflowConst)}, "
                  "BC_type nn_constant), and the 1D solver uses their value as Qin")
    else:
        inflowFunc = 'NA'
        if inflow_func_1d:
            print(f"generate1DPythonSimInitFile :: WARNING: inflow_func_1d = {inflow_func_1d} of the user inputs is not "
                  "used: the 1D network has NO inflow BC (inlet_flow module at the inlet of a 1D vessel)")

    # values of the constant inflow (code 0, constant inlet_flow module), inlet pressure (code 1: inlet_pressure module),
    # outlet pressure (code -1: outlet_pressure module) and outflow (code -2: outlet_flow module) BCs: the constant
    # values of the modules, i.e. their parameters P_<module> (pressure, [J/m3]) and v_<module> (flow, [m3/s]). The 1D
    # solver takes one value for all the BCs of each kind (Qin, Pin, Pout, Qout of [BCs]): if the modules of one kind
    # have different values, the first one is used, with a warning.
    bcKinds = {}
    if inflowFunc=='constant':
        bcKinds[0] = ('Qin', 'inlet', 'v', 1.0e+6, 'cm3/s', 'inflow') # [m3/s] -> [cm3/s]
    bcKinds.update({1: ('Pin', 'inlet', 'P', 1.0e+1, 'dyne/cm2', 'inlet pressure'), # [J/m3] -> [dyne/cm2]
                    -1: ('Pout', 'outlet', 'P', 1.0e+1, 'dyne/cm2', 'outlet pressure'), # [J/m3] -> [dyne/cm2]
                    -2: ('Qout', 'outlet', 'v', 1.0e+6, 'cm3/s', 'outflow')}) # [m3/s] -> [cm3/s]
    bcValues = {} # key of [BCs] -> list of (module name, value in the units of the 1D solver)
    for i in range(len(nodes)):
        if nodes[i]['type'] not in bcKinds:
            continue
        key, side, var, conv, units, kind = bcKinds[nodes[i]['type']]
        nameV = vess[nodes[i]['vess_list'][0]]['name']
        module = (bc_modules_1d or {}).get((nameV, side))
        if module is None:
            sys.exit(f"generate1DPythonSimInitFile :: ERROR: no boundary module found at the {side} of the 1D vessel "
                     f"{nameV}, which has an {kind} BC (node {nodes[i]['id']}, code {nodes[i]['type']})")
        nameP = f"{var}_{module['name']}"
        idxParam = df_params.index[df_params["variable_name"] == nameP].tolist()
        if len(idxParam)!=1:
            sys.exit(f"generate1DPythonSimInitFile :: ERROR: parameter {nameP} of the {module['vessel_type']} module "
                     f"{module['name']} (BC_type {module['BC_type']}, at the {side} of the 1D vessel {nameV}) "
                     f"{'not found' if len(idxParam)==0 else 'found more than once'} in the parameters file: the 1D "
                     f"solver takes its constant value as {key}")
        bcValues.setdefault(key, []).append((module['name'], float(df_params.at[idxParam[0],"value"])*conv))

    config['BCs'] = {
        'nIn': nIn,
        'inflowIdxs': ' '.join(idxs_vIn),
        'nOut': nOut,
        'outflowIdxs': ' '.join(idxs_vOut),
        # code -1 nodes are outlet_pressure modules at the outlet of a 1D vessel (see generate1DPythonModelFiles()):
        # outlet pressure BC, with the pressure Pout below
        'termType': 'pOutflow',
        'PoutFile': 'NA',
        'inflowFunc': inflowFunc # built-in inflow function; constant: constant inflow Qin below; NA: no inflow BC
    }
    for key, units, kind in [(k[0], k[4], k[5]) for k in bcKinds.values()]:
        if key not in bcValues: # no BC of this kind: the 1D solver does not use the value
            continue
        names = [name for (name, value) in bcValues[key]]
        values = [value for (name, value) in bcValues[key]]
        if not all(np.isclose(value, values[0], rtol=1e-12, atol=0.0) for value in values):
            print(f"generate1DPythonSimInitFile :: WARNING: the 1D solver takes one {key} for all its {kind} BCs, but the "
                  f"modules have different values: "
                  f"{', '.join(f'{value:g} {units} ({name})' for (name, value) in bcValues[key])}: using the first one, "
                  f"{values[0]:g} {units} ({names[0]})")
        config['BCs'][key] = f"{values[0]:.15g}"
        print(f"generate1DPythonSimInitFile :: NOTE: {key} = {values[0]:g} {units} for the {kind} BCs of the 1D solver "
              f"({', '.join(names)})")


    # order of the finite volume scheme (FVord) consistent with the ODE solver (checked by the 1d solver):
    # 1 with explEul and with PETSc BDF1, 2 otherwise
    if ODEsolver=='explEul' or (ODEsolver=='PETSC' and petsc_solver=='BDF1'):
        FVord = 1
        ODEord = 1
    else:
        FVord = 2
        if (ODEsolver=='Heun' or ODEsolver=='midpoint'):
            ODEord = 2
        elif ODEsolver=='RK4':
            ODEord = 4
        elif ODEsolver=='PETSC': # CN or BDF2
            ODEord = 2
        else: # ODEsolver = 'CVODE'
            ODEord = -1

    config['numerics'] = {
        'CFL': 0.9,
        'FVord': FVord,
        'slopeType': 'ENO',
        'ODEord': ODEord,
        'ODEsolver': ODEsolver
    }
    if ODEsolver=='PETSC':
        # the same scheme as the 0d solver, checked at start-up of a coupled simulation (see run1DBFmodel.py)
        config['numerics']['PETSc_solver'] = petsc_solver


    config['results'] = {
        'outputFoldName': 'res',
        'tSaveIni': pre_time, # results saved from pre_time, as in the 0d solver
        'saveResSpace': 0,
        'tSaveIniSpace': pre_time # with saveResSpace 1
    }


    MAP = 90.*mmHgDyncm2 # mean arterial pressure[dyne/cm2] (initial pressure for arteries with ICtype=3)
    CVP = 5.*mmHgDyncm2 # central venous pressure [dyne/cm2] (initial pressure for veins with ICtype=3)
    config['calibration'] = {
        'MAP': MAP,
        'CVP': CVP,
        'useImpedance': 1
    }
    # useImpedance=1 to use vessel characteristic impedance to split the total terminal resistance of RCR model,
    # 0 otherwise (some R1/R2 to Rtot ratio is used instead)


    # config['new_section'] = {
    #     'new_key' : 'new_value'
    # }


    print(f'Writing 1D simulation initialisation file {sim_init_file}')
    with open(sim_init_file, 'w') as configfile:
        config.write(configfile)
    print('DONE!')
    print('****')
