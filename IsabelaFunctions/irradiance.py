#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:47:52 2023

@author: oliveira
"""

import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td
from IsabelaFunctions import sun
from tqdm import tqdm


class SatireSSI:
    
    def __init__(self, file, n_days, start_year = 2010, start_month = 6, start_day = 17, wavelengths = [200., 900.]):
        ssi, wl_lower = sun.load_satire_ssi(file, n_days, start_year, start_month, start_day)
        
        if len(wavelengths) > 1:
            selected_ssi = np.zeros((len(wavelengths), ssi.shape[0]))
    
            for w in range(len(wavelengths)):
                selected_ssi[w] = ssi[:, np.where(wl_lower == wavelengths[w])[1][0]] 
        
        else:
            selected_ssi = np.zeros((ssi.shape[0]))
            selected_ssi = ssi[:, np.where(wl_lower == wavelengths)[1][0]] 
            
        self.satire_ssi = selected_ssi
        

class FarSideB:
    
    def __init__(self, path, n_days, input_br = False):
        day0_file = pd.read_csv(path + 'UTC_time_starts_from_dot1.txt', infer_datetime_format = True, names = ['dates'])
        day0 = dt.strptime(day0_file.dates[0], '%Y.%m.%d_%H:%M:%S_TAI').date()
        
        days = []
        for i in range(n_days):
            days.append(day0 + td(days = i))
        
        magnetograms = np.empty((n_days, 181, 360))
        for i in tqdm(range(n_days)):
                magnetograms[i] = np.loadtxt(path + 'CalcMagnetogram.1000.' + str(i))
        
        l0_b0_SDO = pd.read_csv(path + 'l0_b0_SDO.txt', delim_whitespace = True)

        days_SDO = pd.to_datetime(l0_b0_SDO.T_REC, format = '%Y.%m.%d_%H:%M:%S_TAI')
            
        time_stamps = []    
        for i in range(len(days)):
                time_stamps.append(np.where(pd.Timestamp(days[i]) == days_SDO)[0][0])

        l0 = l0_b0_SDO.CRLN_OBS[time_stamps]
        b0 = l0_b0_SDO.CRLT_OBS[time_stamps]

        l0 = np.array(l0)
        b0 = np.array(b0)

        # Substitute the nan for the mean between the lower and higher values
        for i in range(len(l0)):
            if np.isnan(l0[i]) == True:
                l0[i] = np.nanmean(l0[i-1:i+2])

            if np.isnan(b0[i]) == True:
                b0[i] = np.nanmean(b0[i-1:i+2])
        
        self.magnetograms = magnetograms
        self.l0 = l0
        self.b0 = b0
        self.days = days
        
        if input_br:
            data = np.empty_like(magnetograms)
            for i in tqdm(range(n_days)):
                data[i] = np.loadtxt(path + 'input_br/input_br.' + str(i+1))
            self.near_side = data


class Fluxes:
    
    def __init__(self, file):
        wavelengths, angles, flux_quiet, flux_faculae, flux_umbra, flux_penumbra = sun.read_fluxes(file)
        
        self.quiet_sun = sun.compute_interpolations(flux_quiet, angles, len(wavelengths))
        self.faculae = sun.compute_interpolations(flux_faculae, angles, len(wavelengths))
        self.umbra = sun.compute_interpolations(flux_umbra, angles, len(wavelengths))
        self.penumbra = sun.compute_interpolations(flux_penumbra, angles, len(wavelengths))
        self.wavelengths = wavelengths


class ContinuumData:
    
    def __init__(self, path, n_days):
     day0_file = pd.read_csv(path + 'UTC_time_starts_from_dot1_for_IC.txt', infer_datetime_format = True, names = ['dates'])
     day0 = dt.strptime(day0_file.dates[0], '%Y.%m.%d_%H:%M:%S_TAI').date()
     
     days = []
     for i in range(n_days):
         days.append(day0 + td(days = i))
     
     data = np.empty((n_days, 181, 360))
     for i in tqdm(range(n_days)):
             data[i] = np.loadtxt(path + 'obs_ic/observed_ic.' + str(i+1))
     
     l0_b0_SDO = pd.read_csv(path + 'l0_b0_SDO.txt', delim_whitespace = True)

     days_SDO = pd.to_datetime(l0_b0_SDO.T_REC, format = '%Y.%m.%d_%H:%M:%S_TAI')
         
     time_stamps = []    
     for i in range(len(days)):
             time_stamps.append(np.where(pd.Timestamp(days[i]) == days_SDO)[0][0])

     l0 = l0_b0_SDO.CRLN_OBS[time_stamps]
     b0 = l0_b0_SDO.CRLT_OBS[time_stamps]

     l0 = np.array(l0)
     b0 = np.array(b0)

     # Substitute the nan for the mean between the lower and higher values
     for i in range(len(l0)):
         if np.isnan(l0[i]) == True:
             l0[i] = np.nanmean(l0[i-1:i+2])

         if np.isnan(b0[i]) == True:
             b0[i] = np.nanmean(b0[i-1:i+2])
     
     self.continuum = data
     self.l0 = l0
     self.b0 = b0
     self.days = days


class FillingFactors:
     
     def __init__(self, data_mag, sat_fac = 250., min_spot = 60., max_spot = 700.):
         ff_faculae = np.empty_like(data_mag)
         ff_umbra = np.empty_like(data_mag)
         ff_penumbra = np.empty_like(data_mag)

         for i in range(data_mag.shape[0]):
             ff_faculae[i], ff_umbra[i], ff_penumbra[i] = sun.compute_filling_factors(data_mag[i], sat_fac, min_spot, max_spot)

         self.faculae = ff_faculae
         self.umbra = ff_umbra
         self.penumbra = ff_penumbra
         self.spots = ff_umbra + ff_penumbra
         

class FillingFactorsFromFile:
     
        def __init__(self, file_faculae, file_umbra, file_penumbra):
            ff_faculae = np.load(file_faculae)
            ff_umbra = np.load(file_umbra)
            ff_penumbra = np.load(file_penumbra)
    
            self.faculae = ff_faculae
            self.umbra = ff_umbra
            self.penumbra = ff_penumbra
            self.spots = ff_umbra + ff_penumbra