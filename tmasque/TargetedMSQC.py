import pandas as pd
import numpy as np
import re
import itertools
#import pingouin as pg

class TargetedMSQC():
  def __init__(self):
    self.cached = dict()

  def serializeKeyFromChromData(self, chromatogram_data, is_sum):
    ps = chromatogram_data['peptideModifiedSequence']
    fn = chromatogram_data['fileName']
    ts = ','.join(chromatogram_data['transitions'])
    start = chromatogram_data['start']
    end = chromatogram_data['end']
    key = ts + '-' + str(start) + '-' + str(end) + '-' + str(is_sum)
    return ps, fn, key

  def setCachedValue(self, chromatogram_data, func_name, value, is_sum):
    ps, fn, key = self.serializeKeyFromChromData(chromatogram_data, is_sum)
    if ps not in self.cached:
      self.cached[ps] = dict()
    if fn not in self.cached[ps]:
      self.cached[ps][fn] = dict()
    if key not in self.cached[ps][fn]:
      self.cached[ps][fn][key] = dict()
    self.cached[ps][fn][key][func_name] = value
  
  def getCachedValue(self, chromatogram_data, func_name, is_sum):
    ps, fn, key = self.serializeKeyFromChromData(chromatogram_data, is_sum)
    try:
      if ps in self.cached and fn in self.cached[ps] and key in self.cached[ps][fn] and func_name in self.cached[ps][fn][key]:
        return self.cached[ps][fn][key][func_name]
      else:
        return None
    except:
      return None
  
  def clearCachedValue(self, fn, ps):
    try:
      if ps in self.cached and fn in self.cached[ps]:
        self.cached[ps][fn] = dict()
      else:
        return None
    except:
      return None
  
  def resetCachedValue(self):
    self.cached = dict()

  # Intensity(3)
  def _calculateTransitionMaxIntensity(self, chromatogram_data, sum_transition=False):
    key = 'peak_sum_intensity' if sum_transition==True else 'peak_intensity'
    peak_intensity_mx = chromatogram_data[key]
    return pd.DataFrame(peak_intensity_mx.max()).transpose()
  
  def _calculateTransitionMaxBoundaryIntensity(self, chromatogram_data, sum_transition=False):
    key = 'peak_sum_intensity' if sum_transition==True else 'peak_intensity'
    peak_intensity_mx = chromatogram_data[key]
    if len(peak_intensity_mx.index) > 0:
      result = peak_intensity_mx.loc[(0, peak_intensity_mx.index[-1]),:].max()
    else:
      result = pd.DataFrame([], index = peak_intensity_mx.columns)
      result.loc[:, 0] = [np.nan] * len(peak_intensity_mx.columns)
    return pd.DataFrame(result).transpose()

  def calculateTransitionMaxBoundaryIntensityNormalized(self, chromatogram_data, sum_transition=False): #Level 1
    # Set the upperbound value to 5  (the worst case value is 5)
    TransitionMaxBoundaryIntensityNormalized = self._calculateTransitionMaxBoundaryIntensity(chromatogram_data, sum_transition) / self._calculateTransitionMaxIntensity(chromatogram_data, sum_transition)
    TransitionMaxBoundaryIntensityNormalized = TransitionMaxBoundaryIntensityNormalized.replace(np.nan, 5)
    TransitionMaxBoundaryIntensityNormalized.where(TransitionMaxBoundaryIntensityNormalized < 5, 5, inplace=True)
    return TransitionMaxBoundaryIntensityNormalized

  # Shift(4)
  def calculatePairShift(self, chromatogram_data, sum_transition=False): #4 Level 1
    key = 'peak_sum_intensity' if sum_transition==True else 'peak_intensity'
    peak_intensity_mx = chromatogram_data[key]
    peak_time = chromatogram_data['peak_time']
    fragmentIon_list, idx = np.unique(['.'.join(i.split('.')[:-1]) for i in peak_intensity_mx.columns],return_index=True)
    fragmentIon_list = fragmentIon_list[np.argsort(idx)]
    PairShift = []
    for fi in fragmentIon_list:
      if len(peak_intensity_mx[fi+'.light']) == 0 or len(peak_intensity_mx[fi+'.heavy']) == 0 or len(peak_time) <= 1:
        PairShift.append(1)
      else:
        light_max_intensity = max(peak_intensity_mx[fi+'.light'])
        light_max_time = peak_time[peak_intensity_mx[fi+'.light'][peak_intensity_mx[fi+'.light'] == light_max_intensity].index[0]]
        heavy_max_intensity = max(peak_intensity_mx[fi+'.heavy'])
        heavy_max_time = peak_time[peak_intensity_mx[fi+'.heavy'][peak_intensity_mx[fi+'.heavy'] == heavy_max_intensity].index[0]]  
        PairShift.append(round(abs(light_max_time-heavy_max_time) / (peak_time[-1]-peak_time[0]),4))
    PairShift = pd.DataFrame(PairShift).transpose()
    PairShift.columns = fragmentIon_list
    return PairShift

  def _calculateTransitionShiftMatrix(self, chromatogram_data, sum_transition=False): #matrix
    key = 'peak_sum_intensity' if sum_transition==True else 'peak_intensity'
    peak_intensity_mx = chromatogram_data[key]
    peak_time = chromatogram_data['peak_time']
    shift_mx = np.empty((len(peak_intensity_mx.columns),len(peak_intensity_mx.columns)))
    shift_mx[:] = np.nan
    for a,b in list(itertools.combinations(peak_intensity_mx.columns, 2)):
      if len(peak_intensity_mx[a]) == 0 or len(peak_intensity_mx[b]) == 0 or len(peak_time) <= 1:
        shift = 1
      else:
        a_max_intensity = max(peak_intensity_mx[a])
        a_max_time = peak_time[peak_intensity_mx[a][peak_intensity_mx[a] == a_max_intensity].index[0]]
        b_max_intensity = max(peak_intensity_mx[b])
        b_max_time = peak_time[peak_intensity_mx[b][peak_intensity_mx[b] == b_max_intensity].index[0]]
        shift = round(abs(a_max_time-b_max_time) / (peak_time[-1]-peak_time[0]),4)
      shift_mx[list(peak_intensity_mx.columns).index(a)][list(peak_intensity_mx.columns).index(b)] = shift
      shift_mx[list(peak_intensity_mx.columns).index(b)][list(peak_intensity_mx.columns).index(a)] = shift

    all_max_time = []
    for a in peak_intensity_mx.columns:
      if len(peak_intensity_mx[a]) == 0:
        all_max_time.append(0)
      else:
        a_max_intensity = max(peak_intensity_mx[a])
        a_max_time = peak_time[peak_intensity_mx[a][peak_intensity_mx[a] == a_max_intensity].index[0]]    
        all_max_time.append(a_max_time)
    all_max_time = np.array(all_max_time)
    median_max_time = np.median(all_max_time)
    if len(peak_time) >= 2:
      shift_diag = np.round(abs(all_max_time-median_max_time) / (peak_time[-1]-peak_time[0]),4)
    else:
      shift_diag = 1
    np.fill_diagonal(shift_mx, shift_diag)
    shift_mx = pd.DataFrame(shift_mx)
    shift_mx.columns = peak_intensity_mx.columns
    shift_mx.index = peak_intensity_mx.columns
    self.setCachedValue(chromatogram_data, '_calculateTransitionShiftMatrix', shift_mx, sum_transition)
    return shift_mx

  def calculateTransitionShift(self, chromatogram_data, sum_transition=False): #8 Level 1
    TransitionShiftMx = self.getCachedValue(chromatogram_data, '_calculateTransitionShiftMatrix', sum_transition)
    if TransitionShiftMx is None:
      TransitionShiftMx = self._calculateTransitionShiftMatrix(chromatogram_data, sum_transition)
    diag = np.matrix(TransitionShiftMx).diagonal().tolist()[0]
    transition_name = TransitionShiftMx.columns
    TransitionShift_diag = pd.DataFrame(diag).T
    TransitionShift_diag.columns = transition_name
    return TransitionShift_diag

  def calculateIsotopeShift(self, chromatogram_data, sum_transition=False): #2 Level 2
    TransitionShiftMx = self.getCachedValue(chromatogram_data, '_calculateTransitionShiftMatrix', sum_transition)
    if TransitionShiftMx is None:
      TransitionShiftMx = self._calculateTransitionShiftMatrix(chromatogram_data, sum_transition)
    diag = np.matrix(TransitionShiftMx).diagonal().tolist()[0]
    light_shift = np.mean(diag[:int(len(diag)/2)])
    heavy_shift = np.mean(diag[int(len(diag)/2):])
    shift_mx = pd.DataFrame(zip([light_shift],[heavy_shift]))
    shift_mx.columns = ['light','heavy']
    return shift_mx

  def calculatePeakGroupShift(self, chromatogram_data, sum_transition=False): #1 #Level 2
    TransitionShiftMx = self.getCachedValue(chromatogram_data, '_calculateTransitionShiftMatrix', sum_transition)
    if TransitionShiftMx is None:
      TransitionShiftMx = self._calculateTransitionShiftMatrix(chromatogram_data, sum_transition)
    diag = np.matrix(TransitionShiftMx).diagonal().tolist()[0]
    return np.mean(diag)
  
  # Jaggedness(3)
  def calculateTransitionJaggedness(self, chromatogram_data, sum_transition=False, flatness_factor = 0.05): #8 Level 1
    key = 'peak_sum_intensity' if sum_transition==True else 'peak_intensity'
    peak_intensity_mx = chromatogram_data[key]
    peak_diff_mx = pd.DataFrame(np.diff(peak_intensity_mx,axis=0),columns=peak_intensity_mx.columns)
    peak_diff_mx[abs(peak_diff_mx) < flatness_factor*abs(peak_intensity_mx).max()] = 0
    jaggedness = ((abs(pd.DataFrame(np.diff(np.sign(peak_diff_mx),axis=0),columns=peak_intensity_mx.columns)) > 1).sum() - 1) / (len(peak_diff_mx)-1)
    jaggedness = pd.DataFrame(jaggedness).transpose()
    jaggedness.loc[jaggedness.shape[0]] = [0]*jaggedness.shape[1]
    jaggedness = pd.DataFrame(jaggedness.max()).transpose()
    jaggedness = np.round(jaggedness,4)
    self.setCachedValue(chromatogram_data, 'calculateTransitionJaggedness', jaggedness, sum_transition)
    return jaggedness

  def calculateIsotopeJaggedness(self, chromatogram_data, sum_transition=False, flatness_factor = 0.05): #2 #Level 2
    jaggedness = self.getCachedValue(chromatogram_data, 'calculateTransitionJaggedness', sum_transition)
    if jaggedness is None:
      jaggedness = self.calculateTransitionJaggedness(chromatogram_data, sum_transition, flatness_factor)
    r = re.compile(".*light")
    col_name_light = list(filter(r.match,jaggedness.columns))
    jaggedness_light = np.mean(jaggedness[col_name_light].loc[0])
    r = re.compile(".*heavy")
    col_name_heavy = list(filter(r.match,jaggedness.columns))
    jaggedness_heavy = np.mean(jaggedness[col_name_heavy].loc[0])
    jaggedness_mx = pd.DataFrame(zip([jaggedness_light],[jaggedness_heavy]))
    jaggedness_mx.columns = ['light','heavy']
    return jaggedness_mx

  def calculatePeakGroupJaggedness(self, chromatogram_data, sum_transition=False, flatness_factor = 0.05): #1 #Level 2
    jaggedness = self.getCachedValue(chromatogram_data, 'calculateTransitionJaggedness', sum_transition)
    if jaggedness is None:
      jaggedness = self.calculateTransitionJaggedness(chromatogram_data, sum_transition, flatness_factor)
    return  np.mean(jaggedness.loc[0])

  # Similarity(3)
  def _calculatePairSimilarity(self, chromatogram_data, sum_transition=False): #matrix
    key = 'peak_sum_intensity' if sum_transition==True else 'peak_intensity'
    peak_intensity_mx = chromatogram_data[key]
    # if len(peak_intensity_mx) >= 2 and len(peak_intensity_mx.columns[peak_intensity_mx.nunique() <= 1]) == 0:
    #   similarity_mx = peak_intensity_mx.rcorr(method='pearson', decimals=7, padjust='holm',stars=False)
    # else:
    similarity_mx = peak_intensity_mx.corr(method='pearson')
    col_name = similarity_mx.columns
    similarity_mx = np.matrix(similarity_mx)
    np.fill_diagonal(similarity_mx,'0')
    similarity_mx = similarity_mx.astype('float')
    # if len(peak_intensity_mx) >= 2 and len(peak_intensity_mx.columns[peak_intensity_mx.nunique() <= 1]) == 0: # rcorr only generate correlation values in the lower triangle. making symmetry here
    #   similarity_mx = similarity_mx + similarity_mx.T
    np.fill_diagonal(similarity_mx,1)
    similarity_mx = pd.DataFrame(similarity_mx)
    similarity_mx.columns = col_name
    similarity_mx.index = col_name
    PairSimilarity = similarity_mx.replace(np.nan, 0)
    self.setCachedValue(chromatogram_data, '_calculatePairSimilarity', PairSimilarity, sum_transition)
    return PairSimilarity

  def calculatePairSimilarity(self, chromatogram_data, sum_transition=False): #4 Level 1
    similarity_mx = self.getCachedValue(chromatogram_data, '_calculatePairSimilarity', sum_transition)
    if similarity_mx is None:
      similarity_mx = self._calculatePairSimilarity(chromatogram_data, sum_transition)
    col_name = similarity_mx.columns
    pair_name, idx  = np.unique(['.'.join(i.split('.')[:-1]) for i in col_name],return_index=True)
    pair_name = pair_name[np.argsort(idx)]
    PairSimilarity = []
    for i in pair_name:
      PairSimilarity.append(similarity_mx[i+'.light'].loc[i+'.heavy'])
    PairSimilarity = pd.DataFrame(PairSimilarity).T
    PairSimilarity.columns = pair_name    
    return PairSimilarity

  def calculateIsotopeSimilarity(self, chromatogram_data, sum_transition=False): #2 #Level 2
    similarity_mx = self.getCachedValue(chromatogram_data, '_calculatePairSimilarity', sum_transition)
    if similarity_mx is None:
      similarity_mx = self._calculatePairSimilarity(chromatogram_data, sum_transition)
    similarity_light = np.mean(np.matrix(similarity_mx.filter(like='light',axis=0).filter(like='light',axis=1)))
    similarity_heavy = np.mean(np.matrix(similarity_mx.filter(like='heavy',axis=0).filter(like='heavy',axis=1)))
    IsotopeSimilarity = pd.DataFrame(zip([similarity_light],[similarity_heavy]))
    IsotopeSimilarity.columns = ['light','heavy']
    return IsotopeSimilarity

  def calculatePeakGroupSimilarity(self, chromatogram_data, sum_transition=False): #1 #Level 2
    similarity_mx = self.getCachedValue(chromatogram_data, '_calculatePairSimilarity', sum_transition)
    if similarity_mx is None:
      similarity_mx = self._calculatePairSimilarity(chromatogram_data, sum_transition)
    PeakGroupSimilarity = np.mean(np.matrix(similarity_mx))
    return PeakGroupSimilarity
  
  # Symmetry(3)
  def calculateTransitionSymmetry(self, chromatogram_data, sum_transition=False): #8 #Level 1
    key = 'peak_sum_intensity' if sum_transition==True else 'peak_intensity'
    peak_intensity_mx = chromatogram_data[key]
    left = peak_intensity_mx.loc[:int(np.floor(len(peak_intensity_mx)/2))-1].reset_index(drop=True)
    right = peak_intensity_mx.loc[:int(np.floor(len(peak_intensity_mx)/2))+1:-1].reset_index(drop=True)
    TransitionSymmetry = pd.DataFrame(left.corrwith(right, axis = 0).replace(np.nan,0)).transpose()
    self.setCachedValue(chromatogram_data, 'calculateTransitionSymmetry', TransitionSymmetry, sum_transition)
    return TransitionSymmetry

  def calculateIsotopeSymmetry(self, chromatogram_data, sum_transition=False): #2 #Level 2
    symmetry = self.getCachedValue(chromatogram_data, 'calculateTransitionSymmetry', sum_transition)
    if symmetry is None:
      symmetry = self.calculateTransitionSymmetry(chromatogram_data, sum_transition)
    r = re.compile(".*light")
    col_name_light = list(filter(r.match,symmetry.columns))
    symmetry_light = np.mean(symmetry[col_name_light].loc[0])
    r = re.compile(".*heavy")
    col_name_heavy = list(filter(r.match,symmetry.columns))
    symmetry_heavy = np.mean(symmetry[col_name_heavy].loc[0])
    symmetry_mx = pd.DataFrame(zip([symmetry_light],[symmetry_heavy]))
    symmetry_mx.columns = ['light','heavy']
    return symmetry_mx

  def calculatePeakGroupSymmetry(self, chromatogram_data, sum_transition=False): #1 #Level 2
    symmetry = self.getCachedValue(chromatogram_data, 'calculateTransitionSymmetry', sum_transition)
    if symmetry is None:
      symmetry = self.calculateTransitionSymmetry(chromatogram_data, sum_transition)
    return  np.mean(symmetry.loc[0])

  # FWHM(8)
  def _calc_fwhm(self, sig, time):
    peakmax = max(sig)
    try:
      left_index = (sig[(sig - peakmax/2) > 0].index[0]-1, sig[(sig - peakmax/2) > 0].index[0])
    except:
      left_index = (np.nan,np.nan)
    try:
      right_index = (sig[(sig - peakmax/2) > 0].index[-1], sig[(sig - peakmax/2) > 0].index[-1]+1)
    except:
      right_index = (np.nan,np.nan)
        
    if (left_index[0] == -1) or (np.isnan(left_index[0])):
      t_left = time[0]
    else:
      t_left = (time[left_index[1]] - time[left_index[0]])/(sig[left_index[1]] - sig[left_index[0]])*(peakmax/2 - sig[left_index[0]]) + time[left_index[0]]

    if (right_index[1] > (len(time)-1)) or (np.isnan(right_index[1])):
      t_right = time[-1]
    else:
      t_right = (time[right_index[1]] - time[right_index[0]])/(sig[right_index[1]] - sig[right_index[0]])*(peakmax/2 - sig[right_index[0]]) + time[right_index[0]]
    fwhm = t_right - t_left
    return fwhm

  def calculateTransitionFWHM(self, chromatogram_data, sum_transition=False): #8
    key = 'peak_sum_intensity' if sum_transition==True else 'peak_intensity'
    peak_intensity_mx = chromatogram_data.get(key)
    time = chromatogram_data.get('peak_time')
    TransitionFWHM = []
    for col in peak_intensity_mx.columns:
      if len(peak_intensity_mx[col]) == 0 or len(time) <= 1:
        TransitionFWHM.append(5)
      else:
        TransitionFWHM.append(self._calc_fwhm(peak_intensity_mx[col],time))
    TransitionFWHM = pd.DataFrame(TransitionFWHM).transpose()
    TransitionFWHM.columns = peak_intensity_mx.columns
    # Set the upperbound value to 5
    TransitionFWHM.where(TransitionFWHM < 5, 5, inplace=True)
    self.setCachedValue(chromatogram_data, 'calculateTransitionFWHM', TransitionFWHM, sum_transition)
    return TransitionFWHM

  def calculateTransitionFWHM2base(self, chromatogram_data, sum_transition=False): #8 #Level 1
    TransitionFWHM = self.getCachedValue(chromatogram_data, 'calculateTransitionFWHM', sum_transition)
    if TransitionFWHM is None:
      TransitionFWHM = self.calculateTransitionFWHM(chromatogram_data, sum_transition)
    time = chromatogram_data.get('peak_time')
    if len(time) >= 2:
      TransitionFWHM2Base = TransitionFWHM/(time[-1] - time[0])
    else:
      TransitionFWHM2Base = TransitionFWHM
    # Set the upperbound value to 5
    TransitionFWHM2Base.where(TransitionFWHM2Base < 5, 5, inplace=True)
    self.setCachedValue(chromatogram_data, 'calculateTransitionFWHM2base', TransitionFWHM2Base, sum_transition)
    return TransitionFWHM2Base

  def calculateIsotopeFWHM(self, chromatogram_data, sum_transition=False): #2 #Level 2
    FWHM = self.getCachedValue(chromatogram_data, 'calculateTransitionFWHM', sum_transition)
    if FWHM is None:
      FWHM = self.calculateTransitionFWHM(chromatogram_data, sum_transition)
    r = re.compile(".*light")
    col_name_light = list(filter(r.match,FWHM.columns))
    FWHM_light = np.mean(FWHM[col_name_light].loc[0])
    r = re.compile(".*heavy")
    col_name_heavy = list(filter(r.match,FWHM.columns))
    FWHM_heavy = np.mean(FWHM[col_name_heavy].loc[0])
    FWHM_mx = pd.DataFrame(zip([FWHM_light],[FWHM_heavy]))
    FWHM_mx.columns = ['light','heavy']
     # Set the upperbound value to 5
    FWHM_mx.where(FWHM_mx < 5, 5, inplace=True)
    return FWHM_mx

  def calculateIsotopeFWHM2base(self, chromatogram_data, sum_transition=False): #2 Level 2
    FWHM = self.getCachedValue(chromatogram_data, 'calculateTransitionFWHM2base', sum_transition)
    if FWHM is None:
      FWHM = self.calculateTransitionFWHM2base(chromatogram_data, sum_transition)
    r = re.compile(".*light")
    col_name_light = list(filter(r.match,FWHM.columns))
    FWHM_light = np.mean(FWHM[col_name_light].loc[0])
    r = re.compile(".*heavy")
    col_name_heavy = list(filter(r.match,FWHM.columns))
    FWHM_heavy = np.mean(FWHM[col_name_heavy].loc[0])
    FWHM_mx = pd.DataFrame(zip([FWHM_light],[FWHM_heavy]))
    FWHM_mx.columns = ['light','heavy']
     # Set the upperbound value to 5
    FWHM_mx.where(FWHM_mx < 5, 5, inplace=True)
    return FWHM_mx

  def calculatePeakGroupFWHM(self, chromatogram_data, sum_transition=False): #1 #Level 2
    FWHM = self.getCachedValue(chromatogram_data, 'calculateTransitionFWHM', sum_transition)
    if FWHM is None:
      FWHM = self.calculateTransitionFWHM(chromatogram_data, sum_transition)
    PeakGroupFWHM = np.mean(FWHM.loc[0])
    if PeakGroupFWHM > 5:
      PeakGroupFWHM = 5
    return PeakGroupFWHM

  def calculatePeakGroupFWHM2base(self, chromatogram_data, sum_transition=False): #1 #Level 2
    FWHM = self.getCachedValue(chromatogram_data, 'calculateTransitionFWHM2base', sum_transition)
    if FWHM is None:
      FWHM = self.calculateTransitionFWHM2base(chromatogram_data, sum_transition)
    PeakGroupFWHM2base = np.mean(FWHM.loc[0])
    if PeakGroupFWHM2base > 5:
      PeakGroupFWHM2base = 5
    return PeakGroupFWHM2base

  def calculatePairFWHMConsistency(self, chromatogram_data, sum_transition=False): #4 Level 1
    FWHM = self.getCachedValue(chromatogram_data, 'calculateTransitionFWHM', sum_transition)
    if FWHM is None:
      FWHM = self.calculateTransitionFWHM(chromatogram_data, sum_transition)
    cols = list(set(['.'.join(i.split('.')[:-1]) for i in FWHM.columns]))
    PairFWHMConsistency = []
    for c in cols:
      if len(FWHM[c+'.light']) == 0 or len(FWHM[c+'.heavy']) == 0 or FWHM[c+'.heavy'][0] == 0:
        PairFWHMConsistency.append(5)
      else:
        PairFWHMConsistency.append(abs(FWHM[c+'.light'][0]-FWHM[c+'.heavy'][0])/FWHM[c+'.heavy'][0])
    PairFWHMConsistency = pd.DataFrame(PairFWHMConsistency).transpose()
    PairFWHMConsistency.columns = cols
    # set the upperbound value to 5
    PairFWHMConsistency.where(PairFWHMConsistency < 5, 5, inplace=True)
    return PairFWHMConsistency

  def calculateMeanIsotopeFWHMConsistency(self, chrom_data_list, sum_transition=False): #8 # Level 3
    TransitionFWHM_crossAll = pd.DataFrame([])
    rownames = []
    for chrom_data in chrom_data_list:
      if chrom_data is None:
        continue
      eachTransitionFWHM = self.getCachedValue(chrom_data, 'calculateTransitionFWHM', sum_transition)
      if eachTransitionFWHM is None:
        eachTransitionFWHM = self.calculateTransitionFWHM(chrom_data, sum_transition)
      TransitionFWHM_crossAll = pd.concat([TransitionFWHM_crossAll, eachTransitionFWHM])
      rownames.append(chrom_data['fileName'])
    mean = np.mean(TransitionFWHM_crossAll, axis=0)
    TransitionFWHM_crossAll.index = rownames
    MeanIsotopeFWHMConsistency = abs(TransitionFWHM_crossAll - mean) / mean
    # set the upperbound value to 5
    MeanIsotopeFWHMConsistency.where(MeanIsotopeFWHMConsistency < 5, 5, inplace=True)
    return MeanIsotopeFWHMConsistency
  
  #Modality(3)
  def calculateTransitionModality(self, chromatogram_data, sum_transition=False, flatness_factor=0.05): #8 #Level 1
    key = 'peak_sum_intensity' if sum_transition == True else 'peak_intensity'
    peak_intensity_mx = chromatogram_data.get(key)
    # find the differential of the peak    
    peak_diff_mx = pd.DataFrame(np.diff(peak_intensity_mx,axis=0),columns=peak_intensity_mx.columns)
    # any differences that are below the flatnessfactor of the maximum peak height are flattened.
    peak_diff_mx[abs(peak_diff_mx) < flatness_factor*abs(peak_intensity_mx).max()] = 0   
    # find the first and last timepoint where the differential changes sign
    first_fall = []
    last_rise = []
    for col_name in peak_diff_mx.columns:
      if len(peak_diff_mx[col_name][peak_diff_mx[col_name]<0]) > 0:
        first_fall.append(peak_diff_mx[col_name][peak_diff_mx[col_name]<0].index[0])
      else:
        first_fall.append(len(peak_intensity_mx[col_name]))
      if len(peak_diff_mx[col_name][peak_diff_mx[col_name]>0]) > 0:    
        last_rise.append(peak_diff_mx[col_name][peak_diff_mx[col_name]>0].index[-1])
      else:
        last_rise.append(-1)
      # if first fall is after last rise, peak cannot be bi or multi-modal, so max.dip is set to 0. Otherwise it is set to the largest fall or rise between the first fall and last rise
    TransitionModality = []
    for f,l,col_name in zip(first_fall,last_rise,peak_diff_mx.columns):
      max_dip = 0
      if f < l:
        max_dip = max(abs(peak_diff_mx[col_name][list(range(f,l+1,1))]))

      # The output is the maximum dip normalized by the peak height
      if len(peak_intensity_mx[col_name]) == 0:
        modality = 1
      elif max(peak_intensity_mx[col_name])==0:
        # The original paper used 0 as the imputation, we would like to consider this as the worst case. So we gave the worst modality 1.
        modality = 1
      else:
        modality = max_dip/max(peak_intensity_mx[col_name])
      TransitionModality.append(modality)  
    TransitionModality = pd.DataFrame(TransitionModality).transpose()
    TransitionModality.columns = peak_intensity_mx.columns
    self.setCachedValue(chromatogram_data, 'calculateTransitionModality', TransitionModality, sum_transition)
    return TransitionModality

  def calculateIsotopeModality(self, chromatogram_data, sum_transition=False, flatness_factor=0.05): #2 #Level 2
    Modality = self.getCachedValue(chromatogram_data, 'calculateTransitionModality', sum_transition)
    if Modality is None:
      Modality = self.calculateTransitionModality(chromatogram_data, sum_transition)
    r = re.compile(".*light")
    col_name_light = list(filter(r.match,Modality.columns))
    Modality_light = np.mean(Modality[col_name_light].loc[0])
    r = re.compile(".*heavy")
    col_name_heavy = list(filter(r.match,Modality.columns))
    Modality_heavy = np.mean(Modality[col_name_heavy].loc[0])
    IsotopeModality = pd.DataFrame(zip([Modality_light],[Modality_heavy]))
    IsotopeModality.columns = ['light','heavy']
    return IsotopeModality

  def calculatePeakGroupModality(self, chromatogram_data, sum_transition=False, flatness_factor=0.05): #1 #Level 2
    Modality = self.getCachedValue(chromatogram_data, 'calculateTransitionModality', sum_transition)
    if Modality is None:
      Modality = self.calculateTransitionModality(chromatogram_data, sum_transition)
    PeakGroupModality = np.mean(Modality.loc[0])
    return PeakGroupModality
  
  # Area Ratio(4)
  def calculateArea2SumRatioCV(self, chrom_data_list): #8(sum補0) # Level 3
    Area2SumRatio = pd.DataFrame([])
    rownames = []
    for chrom_data in chrom_data_list:
      if chrom_data is None:
        continue
      # Area2SumRatio = Area2SumRatio.append(chrom_data['Area2SumRatio'])
      Area2SumRatio = pd.concat([Area2SumRatio, chrom_data['Area2SumRatio']])
      rownames.append(chrom_data['fileName'])
    Area2SumRatio.index = rownames
    Area2SumRatioCV = pd.DataFrame( np.std(Area2SumRatio,ddof=1) / np.mean(Area2SumRatio, axis=0) ).T
    # Set the upperbound value to 5
    Area2SumRatioCV.where(Area2SumRatioCV < 5, 5, inplace=True)
    return Area2SumRatioCV

  def calculatePeakGroupRatioCorr(self, chromatogram_data): #1(sum補相同數值) #Level 2
    Area2SumRatio = chromatogram_data['Area2SumRatio']
    FragmentIon = list(set(['.'.join(i.split('.')[:-1]) for i in Area2SumRatio.columns]))
    light_col = [fi+'.light' for fi in FragmentIon]
    heavy_col =[fi+'.heavy' for fi in FragmentIon]
    if len(Area2SumRatio[light_col].loc[0]) < 2 or len(Area2SumRatio[heavy_col].loc[0]) < 2:
      return 0
    else:
      corr = np.corrcoef(Area2SumRatio[light_col].loc[0], Area2SumRatio[heavy_col].loc[0])[0][1]
    if np.isnan(corr):
      return 0
    else:
      return corr

  def calculateEachPairRatioConsistency(self, chromatogram_data): #4(sum補0) #Level 1
    Area2SumRatio = chromatogram_data['Area2SumRatio']
    cols = list(set(['.'.join(i.split('.')[:-1]) for i in Area2SumRatio.columns]))
    PairRatioConsistency = []
    for c in cols:
      if Area2SumRatio[c+'.heavy'][0] == 0:
        PairRatioConsistency.append(5)
      else:
        PairRatioConsistency.append(abs(Area2SumRatio[c+'.light'][0] - Area2SumRatio[c+'.heavy'][0]) / Area2SumRatio[c+'.heavy'][0])
    PairRatioConsistency = pd.DataFrame(PairRatioConsistency).transpose()
    PairRatioConsistency.columns = cols
    # set the upperbound value to 5
    PairRatioConsistency.where(PairRatioConsistency < 5, 5, inplace=True)
    return PairRatioConsistency

  def calculatePairRatioConsistency(self, chrom_data_list):
    rownames = []
    PairRatioConsistency = pd.DataFrame([])
    for chrom_data in chrom_data_list:
      if chrom_data is None:
        continue
      PairRatioConsistency = pd.concat([PairRatioConsistency, self.calculateEachPairRatioConsistency(chrom_data)])
      rownames.append(chrom_data['fileName'])
    PairRatioConsistency.index = rownames
    return PairRatioConsistency


  def calculateMeanIsotopeRatioConsistency(self, chrom_data_list): #8(sum補0) #Level 3, each File has its own value
    rownames = []
    Area2SumRatio = pd.DataFrame([])
    for chrom_data in chrom_data_list:
      if chrom_data is None:
        continue
      Area2SumRatio = pd.concat([Area2SumRatio, chrom_data['Area2SumRatio']])
      rownames.append(chrom_data['fileName'])
    mean = np.mean(Area2SumRatio, axis=0)
    Area2SumRatio.index = rownames
    MeanIsotopeRatioConsistency = abs(Area2SumRatio - mean) / mean
     # Set the upperbound value to 5
    MeanIsotopeRatioConsistency.where(MeanIsotopeRatioConsistency < 5, 5, inplace=True)
    return MeanIsotopeRatioConsistency

  # RT(1)
  # PeakCenter = (MaxEndTime + MinStartTime)/2
  # MeanIsotopeRTConsistency = abs(PeakCenter - MeanPeakCenter)/MeanPeakCenter
  # MeanPeakCenter: mean of PeakCenter across all samples is calculated for each transition and isotope label.
  def calculateMeanIsotopeRTConsistency(self, chrom_data_list): #1(sum補相同數值) # Level 3
    # The chromatograms of the same filename have the same value.
    peakRT_crossAll = list()
    rownames = []
    for chrom_data in chrom_data_list:
      if chrom_data is None:
        continue
      peakRT_crossAll.append((chrom_data['start'] + chrom_data['end'])/2)
      rownames.append(chrom_data['fileName'])
    if len(peakRT_crossAll) == 0:
      return pd.DataFrame([])
    mean = np.mean(peakRT_crossAll)
    meanIsotopeRTconsistency = pd.DataFrame(abs(peakRT_crossAll - mean) / mean)
    meanIsotopeRTconsistency.index = rownames
    # Set the upperbound value to 5
    meanIsotopeRTconsistency.where(meanIsotopeRTconsistency < 5, 5, inplace=True)
    return meanIsotopeRTconsistency

