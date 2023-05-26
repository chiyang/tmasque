import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from multiprocessing import Pool
import time
from .TargetedMSQC import TargetedMSQC
from .Chromatogram import Chromatogram

class PeakQualityControl():
  def __init__(self, chromatogramFile, boundaryFile, core_num=1, switchLightHeavy=False):
    self.ms_data = Chromatogram(chromatogramFile, boundaryFile, switchLightHeavy=switchLightHeavy)
    self.TargetedMSQC = TargetedMSQC()
    self.filename_list = np.unique(self.ms_data.chrom['FileName'].astype(str))
    self.peptide_list = np.unique(self.ms_data.chrom['PeptideModifiedSequence'].astype(str))
    self.chrom_dict = dict()
    self.core_num = core_num
    self.quality_indexes = dict()
    self.cached = dict()
    self._isSwitchLightHeavy = switchLightHeavy
  
  def isSwitchLightHeavy(self):
    return self._isSwitchLightHeavy

  def _saveQualityMetrics(self, fn, ps, level, qualityMetricValues, isotope, transition):
    if fn not in self.quality_indexes:
      self.quality_indexes[fn] = dict()
    if ps not in self.quality_indexes[fn]:
      self.quality_indexes[fn][ps] = dict()
    if level not in self.quality_indexes[fn][ps]:
      self.quality_indexes[fn][ps][level] = dict()
    if level == 'transition_level':
      if isotope not in self.quality_indexes[fn][ps][level]:
        self.quality_indexes[fn][ps][level][isotope] = dict()
      self.quality_indexes[fn][ps]['transition_level'][isotope][transition] = qualityMetricValues
    elif level == 'transition_pair_level':
      self.quality_indexes[fn][ps]['transition_pair_level'][transition] = qualityMetricValues
    elif level == 'isotope_level':
      self.quality_indexes[fn][ps]['isotope_level'][isotope] = qualityMetricValues
    elif level == 'peak_group_level':
      self.quality_indexes[fn][ps]['peak_group_level'] = qualityMetricValues
  
    
  def selectChromData(self, fileName, pepModSeq, transitions=None):
    if transitions is not None:
      try:
        return self.ms_data.getChromData(fileName, pepModSeq, transitions = transitions)
      except Exception as e:
        print(e)
        return None
    if fileName in self.chrom_dict and pepModSeq in self.chrom_dict[fileName]:
      # Using cached data
      chromData = self.chrom_dict[fileName][pepModSeq]
    else:
      try:
        chromData = self.ms_data.getChromData(fileName, pepModSeq)
        if fileName not in self.chrom_dict:
          self.chrom_dict[fileName] = dict()
        self.chrom_dict[fileName][pepModSeq] = chromData
      except:
        chromData = None
    return chromData

  def _interpolate_intensity(self, col_data, interpolated_rt, original_rt):
    return np.interp(interpolated_rt, original_rt, col_data, left=0, right=0)

  def interpolate_chrom_data(self, chrom_data, time_point_num = 1024, min_rt=None, max_rt=None):
    intensity = chrom_data['intensity']
    rt = chrom_data['time']
    if min_rt is None:
      min_rt = rt[0]
    if max_rt is None:
      max_rt = rt[-1]
    interpolated_rt = np.linspace(min_rt, max_rt, time_point_num)
    interpolated_intensity = intensity.apply(func = self._interpolate_intensity, axis=0, args=[interpolated_rt, rt])
    return self.ms_data.make_chrom_obj(chrom_data['fileName'], chrom_data['peptideModifiedSequence'], interpolated_rt, interpolated_intensity, chrom_data['start'], chrom_data['end'])

  def get_all_chrom_data_by_peptide(self, peptide):
    valid_chrom_data = []
    for fn in self.filename_list:
      chrom = self.selectChromData(fn, peptide)
      if chrom is not None:
        valid_chrom_data.append(chrom)
    return valid_chrom_data

  def updateBoundary(self, chrom=None, fn=None, ps=None, start=None, end=None):
    try:
      if chrom is None:
        chrom_data = self.ms_data.getChromData(fn, ps, start=start, end=end)
      else:
        chrom_data = chrom
        chrom_data['start'] = start
        chrom_data['end'] = end
        all_time = chrom_data['time']
        all_intensity = chrom_data['intensity']
        peak_filter = (all_time >= start) & (all_time <= end)
        peak_intensity = chrom_data['intensity'][peak_filter].reset_index(drop=True)
        endogenous_cols = chrom_data['endogenous_cols']
        standard_cols = chrom_data['standard_cols']
        peak_time = all_time[peak_filter]
        chrom_data['peak_time'] = peak_time
        chrom_data['peak_intensity'] = peak_intensity
        chrom_data['peak_sum_intensity'] = pd.DataFrame(zip(peak_intensity[endogenous_cols].sum(1),peak_intensity[standard_cols].sum(1)),columns=['sum.light','sum.heavy'])
        area = []
        for c in peak_intensity.columns:
          area.append(np.trapz(peak_intensity[c], peak_time))
        area = pd.DataFrame(area).transpose()
        area.columns = peak_intensity.columns

        Area2SumRatio_endo = area[endogenous_cols] / sum(area[endogenous_cols].loc[0]) #Area2SumRatio要分開light Heavy
        Area2SumRatio_stand = area[standard_cols]/sum(area[standard_cols].loc[0]) #Area2SumRatio要分開light Heavy
        chrom_data['Area2SumRatio'] = pd.concat([Area2SumRatio_endo, Area2SumRatio_stand], axis=1)
        chrom_data['area'] = area
      if chrom_data is None:
        return None
      fn = chrom_data['fileName']
      ps = chrom_data['peptideModifiedSequence']
      if fn not in self.chrom_dict:
        self.chrom_dict[fn] = dict()
      self.chrom_dict[fn][ps] = chrom_data
      self.TargetedMSQC.clearCachedValue(fn, ps)
    except Exception as e:
      print(e)
      print('Error')
      chrom_data = None
    return chrom_data
  
  def plotChromData(self, chrom = None, fn=None, ps=None, transitions=None):
    if chrom is None:
      chrom_data = self.selectChromData(fn, ps, transitions=transitions)
    else:
      chrom_data = chrom
    if chrom_data is None:
      return None
    fn = chrom_data['fileName']
    ps = chrom_data['peptideModifiedSequence']
    fig, (ax1, ax2) = plt.subplots(2, 1)
    time = chrom_data['time']
    light_global_max_intensity = chrom_data['intensity'][chrom_data['endogenous_cols']].max().max()
    heavy_global_max_intensity = chrom_data['intensity'][chrom_data['standard_cols']].max().max()
    light_exp = len(str(int(light_global_max_intensity))) - 2
    heavy_exp = len(str(int(heavy_global_max_intensity))) - 2
    if light_exp < 0 :
      light_exp = 0
    if heavy_exp < 0:
      heavy_exp = 0
    for fragment in chrom_data['transitions']:
      ax1.plot(time, [intensity/(10**light_exp) for intensity in chrom_data['intensity'][fragment + '.light']],  linewidth=1, label=fragment + '.light')
      ax2.plot(time, [intensity/(10**heavy_exp) for intensity in chrom_data['intensity'][fragment + '.heavy']],  linewidth=1, label=fragment + '.heavy')
    ax1.set_title(fn + ': ' + ps, fontsize=8)
    ax1.set_ylabel('Intensity(10^%s)'%light_exp)
    ax1.axvline(chrom_data['start'], linestyle= '--', linewidth=1,color='black')
    ax1.axvline(chrom_data['end'], linestyle= '--', linewidth=1,color='black')
    ax1.legend(fontsize=8,loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    
    ax2.set_ylabel('Intensity(10^%s)'%heavy_exp)
    ax2.set_xlabel('Retention Time')
    ax2.axvline(chrom_data['start'], linestyle= '--', linewidth=1,color='black')
    ax2.axvline(chrom_data['end'], linestyle= '--', linewidth=1,color='black')
    ax2.legend(fontsize=8,loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    return fig

# ----------------Export quality indexes-------------------------------------------------
  def _getType1QualityForModPepSeq(self, ps):
    transition_pair_level = []
    transition_level = []
    for fn in self.filename_list:
      try:
        chrom_data = self.selectChromData(fn, ps)
      except:
        chrom_data = None
      if chrom_data is None:
        continue
      try:
        #for sum_transition_per_isotope in [False, True]:
        # Transition pair level
        TransitionPair_Name = np.unique(['.'.join(i.split('.')[:-1]) for i in chrom_data.get('peak_intensity').columns])
        # PairRatioConsistency = self.TargetedMSQC.calculatePairRatioConsistency(chrom_data)
        PairSimilarity = self.TargetedMSQC.calculatePairSimilarity(chrom_data, sum_transition=False)
        PairShift = self.TargetedMSQC.calculatePairShift(chrom_data, sum_transition=False)
        PairFWHMConsistency = self.TargetedMSQC.calculatePairFWHMConsistency(chrom_data, sum_transition=False)
        for tp in TransitionPair_Name:
          transition_pair_level.append(dict(
            FileName = fn,
            PeptideModifiedSequence = ps,
            TransitionPair = tp,
            # PairRatioConsistency = PairRatioConsistency[tp][0],
            PairSimilarity = PairSimilarity[tp][0],
            PairShift = PairShift[tp][0],
            PairFWHMConsistency = PairFWHMConsistency[tp][0]
          ))
        #Transition level
        Transition_Name = chrom_data.get('peak_intensity').columns
        TransitionJaggedness = self.TargetedMSQC.calculateTransitionJaggedness(chrom_data, sum_transition=False, flatness_factor=0.05) #sum (sum.light | sum.heavy)
        TransitionSymmetry = self.TargetedMSQC.calculateTransitionSymmetry(chrom_data, sum_transition=False) #sum (sum.light | sum.heavy)
        TransitionModality = self.TargetedMSQC.calculateTransitionModality(chrom_data, sum_transition=False, flatness_factor=0.05) #sum (sum.light | sum.heavy)
        TransitionShift = self.TargetedMSQC.calculateTransitionShift(chrom_data, sum_transition=False) #sum (sum.light | sum.heavy)
        TransitionFWHM = self.TargetedMSQC.calculateTransitionFWHM(chrom_data, sum_transition=False) #sum (sum.light | sum.heavy)
        TransitionFWHM2base = self.TargetedMSQC.calculateTransitionFWHM2base(chrom_data, sum_transition=False) #sum (sum.light | sum.heavy)
        TransitionMaxBoundaryIntensityNormalized = self.TargetedMSQC.calculateTransitionMaxBoundaryIntensityNormalized(chrom_data, sum_transition=False) #sum (sum.light | sum.heavy)
        for t in Transition_Name:
          isotope = t.split('.')[-1] # heavy | light
          transition = '.'.join(t.split('.')[0:-1])
          transition_level.append(dict(
            FileName = fn,
            PeptideModifiedSequence = ps,
            Isotope = isotope,
            Transition = transition,
            TransitionJaggedness = TransitionJaggedness[t][0],
            TransitionSymmetry = TransitionSymmetry[t][0],
            TransitionModality = TransitionModality[t][0],
            TransitionShift = TransitionShift[t][0],
            TransitionFWHM = TransitionFWHM[t][0],
            TransitionFWHM2base = TransitionFWHM2base[t][0],
            TransitionMaxBoundaryIntensityNormalized = TransitionMaxBoundaryIntensityNormalized[t][0],
          ))
      except Exception as e:
        print(e)
        continue
    return dict(
      transition_pair_level = transition_pair_level,
      transition_level = transition_level
    )
  def _getType2QualityForModPepSeq(self, ps):
    peak_group_level = []
    isotope_level = []
    for fn in self.filename_list:
      try:
        chrom_data = self.selectChromData(fn, ps)
      except:
        chrom_data = None
      if chrom_data is None:
        continue
      try:
        peak_group_level.append(dict(
          FileName = fn,
          PeptideModifiedSequence = ps,
          PeakGroupRatioCorr = self.TargetedMSQC.calculatePeakGroupRatioCorr(chrom_data),
          PeakGroupJaggedness = self.TargetedMSQC.calculatePeakGroupJaggedness(chrom_data, flatness_factor=0.05),
          PeakGroupSymmetry = self.TargetedMSQC.calculatePeakGroupSymmetry(chrom_data),
          PeakGroupSimilarity = self.TargetedMSQC.calculatePeakGroupSimilarity(chrom_data), 
          PeakGroupModality = self.TargetedMSQC.calculatePeakGroupModality(chrom_data, flatness_factor=0.05), 
          PeakGroupShift = self.TargetedMSQC.calculatePeakGroupShift(chrom_data), 
          PeakGroupFWHM = self.TargetedMSQC.calculatePeakGroupFWHM(chrom_data), 
          PeakGroupFWHM2base = self.TargetedMSQC.calculatePeakGroupFWHM2base(chrom_data)
        ))
        IsotopeJaggedness = self.TargetedMSQC.calculateIsotopeJaggedness(chrom_data, flatness_factor=0.05) 
        IsotopeSymmetry = self.TargetedMSQC.calculateIsotopeSymmetry(chrom_data) 
        IsotopeSimilarity = self.TargetedMSQC.calculateIsotopeSimilarity(chrom_data) 
        IsotopeModality = self.TargetedMSQC.calculateIsotopeModality(chrom_data, flatness_factor=0.05) 
        IsotopeShift = self.TargetedMSQC.calculateIsotopeShift(chrom_data) 
        IsotopeFWHM = self.TargetedMSQC.calculateIsotopeFWHM(chrom_data)
        IsotopeFWHM2base = self.TargetedMSQC.calculateIsotopeFWHM2base(chrom_data)
        for isotope in ['light','heavy']:
          isotope_level.append(dict(
            FileName = fn,
            PeptideModifiedSequence = ps,
            Isotope = isotope,
            IsotopeJaggedness = IsotopeJaggedness[isotope][0],
            IsotopeSymmetry = IsotopeSymmetry[isotope][0],
            IsotopeSimilarity = IsotopeSimilarity[isotope][0],
            IsotopeModality = IsotopeModality[isotope][0],
            IsotopeShift = IsotopeShift[isotope][0],
            IsotopeFWHM = IsotopeFWHM[isotope][0],
            IsotopeFWHM2base = IsotopeFWHM2base[isotope][0]
          ))
      except Exception as e:
        print(e)
        continue
    return dict(
      peak_group_level = peak_group_level,
      isotope_level = isotope_level
    )
  def _getType3QualityForModPepSeq(self, ps):
    level3_rewards = []
    level3_data = self.serializeLevel3(ps, True)
    data_columns = self.serializeLevel3Title()
    for sample, row in level3_data.iterrows():
      level3_rewards.append(dict(
        file = sample,
        peptide = ps,
        transition = row['transition'],
        data = row[data_columns].values.tolist()
      ))
    return level3_rewards

  def calculateQualityType1(self):
    print('Calculating Type 1 Quality (individual transition) ...')
    start = time.time()
    pool = Pool(self.core_num)
    results = pool.map(self._getType1QualityForModPepSeq, self.peptide_list)
    pool.close()
    pool.join()
    end = time.time()
    print('Total calculation time for Type 1 (individual transition): %.2f seconds' % (end - start))
    #print(end - start)
    transition_pair_level = []
    transition_level = []
    serializedData = []
    self.quality_indexes = dict()
    for eachResult in results:
      for eachTransitionPair in eachResult['transition_pair_level']:
        transition_pair_level.append(eachTransitionPair)
      for eachTransition in eachResult['transition_level']:
        transition_level.append(eachTransition)
    #transition_level
    for eachItem in transition_level:
      fn = eachItem['FileName']
      ps = eachItem['PeptideModifiedSequence']
      isotope = eachItem['Isotope']
      transition = eachItem['Transition']
      qualityMetricValues = dict(
        TransitionJaggedness = eachItem['TransitionJaggedness'],
        TransitionSymmetry = eachItem['TransitionSymmetry'],
        TransitionModality = eachItem['TransitionModality'],
        TransitionShift = eachItem['TransitionShift'],
        TransitionFWHM = eachItem['TransitionFWHM'],
        TransitionFWHM2base = eachItem['TransitionFWHM2base'],
        TransitionMaxBoundaryIntensityNormalized = eachItem['TransitionMaxBoundaryIntensityNormalized'],
      )
      self._saveQualityMetrics(fn, ps, 'transition_level', qualityMetricValues, isotope, transition)
    #transition_pair_level
    for eachItem in transition_pair_level:
      fn = eachItem['FileName']
      ps = eachItem['PeptideModifiedSequence']
      tp = eachItem['TransitionPair']
      qualityMetricValues = dict(
        # PairRatioConsistency = eachItem['PairRatioConsistency'],
        PairSimilarity = eachItem['PairSimilarity'],
        PairShift = eachItem['PairShift'],
        PairFWHMConsistency = eachItem['PairFWHMConsistency']
      )
      self._saveQualityMetrics(fn, ps, 'transition_pair_level', qualityMetricValues, None, tp)
      transition_level = self.quality_indexes[fn][ps]['transition_level']
      eachLightTransitionQuality = transition_level['light'][tp]
      eachHeavyTransitionQuality = transition_level['heavy'][tp]
      data = [
        # light transition
        eachLightTransitionQuality['TransitionJaggedness'],
        eachLightTransitionQuality['TransitionSymmetry'],
        eachLightTransitionQuality['TransitionModality'],
        eachLightTransitionQuality['TransitionShift'],
        eachLightTransitionQuality['TransitionFWHM'],
        eachLightTransitionQuality['TransitionFWHM2base'],
        eachLightTransitionQuality['TransitionMaxBoundaryIntensityNormalized'],
        # heavy transition
        eachHeavyTransitionQuality['TransitionJaggedness'],
        eachHeavyTransitionQuality['TransitionSymmetry'],
        eachHeavyTransitionQuality['TransitionModality'],
        eachHeavyTransitionQuality['TransitionShift'],
        eachHeavyTransitionQuality['TransitionFWHM'],
        eachHeavyTransitionQuality['TransitionFWHM2base'],
        eachHeavyTransitionQuality['TransitionMaxBoundaryIntensityNormalized'],
        # Combined
        # eachItem['PairRatioConsistency'],
        eachItem['PairSimilarity'],
        eachItem['PairShift'],
        eachItem['PairFWHMConsistency']
      ]
      if np.isnan(data).any():
        raise ValueError("Quality Metrics Level 1 has nan values (Peptide: " + ps + "; file: " + fn + '; transition: ' + tp + ")")
      serializedData.append({
        'file': fn,
        'peptide': ps,
        'transition': tp,
        'data': data
      })
    # end4 = time.time()
    # print('Storing transition pair level data in %.2f seconds' % (end4 - end3))
    return serializedData

  def calculateQualityType3(self):
    print('Calculating Type 3 Quality (consistency) ...')
    start = time.time()
    pool = Pool(self.core_num)
    results = pool.map(self._getType3QualityForModPepSeq, self.peptide_list)
    pool.close()
    pool.join()
    end = time.time()
    print('Total calculation time for Type 3 Quality (consistency): %.2f seconds' % (end - start))
    self.quality_indexes = dict()
    level3_rewards = []
    for eachResult in results:
      for eachLv3Rewards in eachResult:
        if np.isnan(eachLv3Rewards['data']).any():
          raise ValueError("Quality Metrics Level 3 has nan values (Peptide: " + eachLv3Rewards['peptide'] + "; file: " + eachLv3Rewards['file'] + '; transition: ' + eachLv3Rewards['transition'] + ")")
        level3_rewards.append(eachLv3Rewards)
    return level3_rewards

  def calculateQualityType2(self):
    print('Calculating Type 2 Quality (peak group) ...')
    start = time.time()
    pool = Pool(self.core_num)
    results = pool.map(self._getType2QualityForModPepSeq, self.peptide_list)
    pool.close()
    pool.join()
    end = time.time()
    print('Total calculation time for Type 2 Quality (peak group): %.2f seconds' % (end - start))
    #print(end - start)
    peak_group_level = []
    isotope_level = []
    serializedData = []
    self.quality_indexes = dict()
    for eachResult in results:
      for eachPeakGroup in eachResult['peak_group_level']:
        peak_group_level.append(eachPeakGroup)
      for eachIsotope in eachResult['isotope_level']:
        isotope_level.append(eachIsotope)

    #Isotope level
    for eachItem in isotope_level:
      fn = eachItem['FileName']
      ps = eachItem['PeptideModifiedSequence']
      isotope = eachItem['Isotope']
      qualityMetricValues = dict(
        IsotopeJaggedness = eachItem['IsotopeJaggedness'],
        IsotopeSymmetry = eachItem['IsotopeSymmetry'],
        IsotopeSimilarity = eachItem['IsotopeSimilarity'],
        IsotopeModality = eachItem['IsotopeModality'],
        IsotopeShift = eachItem['IsotopeShift'],
        IsotopeFWHM = eachItem['IsotopeFWHM'],
        IsotopeFWHM2base = eachItem['IsotopeFWHM2base'],
      )
      self._saveQualityMetrics(fn, ps, 'isotope_level', qualityMetricValues, isotope, None)

    #Peak group level
    for eachItem in peak_group_level:
      fn = eachItem['FileName']
      ps = eachItem['PeptideModifiedSequence']
      qualityMetricValues = dict(
        PeakGroupRatioCorr = eachItem['PeakGroupRatioCorr'],
        PeakGroupJaggedness = eachItem['PeakGroupJaggedness'],
        PeakGroupSymmetry = eachItem['PeakGroupSymmetry'],
        PeakGroupSimilarity = eachItem['PeakGroupSimilarity'],
        PeakGroupModality = eachItem['PeakGroupModality'],
        PeakGroupShift = eachItem['PeakGroupShift'],
        PeakGroupFWHM = eachItem['PeakGroupFWHM'],
        PeakGroupFWHM2base = eachItem['PeakGroupFWHM2base']
      )
      self._saveQualityMetrics(fn, ps, 'peak_group_level', qualityMetricValues, None, None)

      isotopeLevelLight = self.quality_indexes[fn][ps]['isotope_level']['light']
      isotopeLevelHeavy = self.quality_indexes[fn][ps]['isotope_level']['heavy']
      data = [
        # Isotope level: light
        isotopeLevelLight['IsotopeJaggedness'],
        isotopeLevelLight['IsotopeSymmetry'],
        isotopeLevelLight['IsotopeSimilarity'],
        isotopeLevelLight['IsotopeModality'],
        isotopeLevelLight['IsotopeShift'],
        isotopeLevelLight['IsotopeFWHM'],
        isotopeLevelLight['IsotopeFWHM2base'],
        # Isotope level: heavy
        isotopeLevelHeavy['IsotopeJaggedness'],
        isotopeLevelHeavy['IsotopeSymmetry'],
        isotopeLevelHeavy['IsotopeSimilarity'],
        isotopeLevelHeavy['IsotopeModality'],
        isotopeLevelHeavy['IsotopeShift'],
        isotopeLevelHeavy['IsotopeFWHM'],
        isotopeLevelHeavy['IsotopeFWHM2base'],
        # Peak group level
        eachItem['PeakGroupRatioCorr'],
        eachItem['PeakGroupJaggedness'],
        eachItem['PeakGroupSymmetry'],
        eachItem['PeakGroupSimilarity'],
        eachItem['PeakGroupModality'],
        eachItem['PeakGroupShift'],
        eachItem['PeakGroupFWHM'],
        eachItem['PeakGroupFWHM2base']
      ]
      if np.isnan(data).any():
        raise ValueError("Quality Metrics Level 2 has nan values (Peptide: " + ps + "; file: " + fn + ")")
      serializedData.append({
        'file': fn,
        'peptide': ps,
        'data': data
      })
    return serializedData

  def serializeLevel1(self, chrom = None, transition=None, fn=None, ps=None, transitions=None):
    if chrom is None:
      chrom_data = self.selectChromData(fn, ps, transitions)
    else:
      chrom_data = chrom
    
    if chrom_data is None:
      return None
    
    #Transition Pair Level
    # PairRatioConsistency = self.TargetedMSQC.calculatePairRatioConsistency(chrom_data)
    PairSimilarity = self.TargetedMSQC.calculatePairSimilarity(chrom_data, sum_transition=False) #sum (sum)
    PairShift = self.TargetedMSQC.calculatePairShift(chrom_data, sum_transition=False) #sum (sum)
    PairFWHMConsistency = self.TargetedMSQC.calculatePairFWHMConsistency(chrom_data, sum_transition=False) #sum (sum)
    #Transition Level
    TransitionJaggedness = self.TargetedMSQC.calculateTransitionJaggedness(chrom_data, sum_transition=False, flatness_factor=0.05) #sum (sum.light | sum.heavy)
    TransitionSymmetry = self.TargetedMSQC.calculateTransitionSymmetry(chrom_data, sum_transition=False) #sum (sum.light | sum.heavy)
    TransitionModality = self.TargetedMSQC.calculateTransitionModality(chrom_data, sum_transition=False, flatness_factor=0.05) #sum (sum.light | sum.heavy)
    TransitionShift = self.TargetedMSQC.calculateTransitionShift(chrom_data, sum_transition=False) #sum (sum.light | sum.heavy)
    TransitionFWHM = self.TargetedMSQC.calculateTransitionFWHM(chrom_data, sum_transition=False) #sum (sum.light | sum.heavy)
    TransitionFWHM2base = self.TargetedMSQC.calculateTransitionFWHM2base(chrom_data, sum_transition=False) #sum (sum.light | sum.heavy)
    TransitionMaxBoundaryIntensityNormalized = self.TargetedMSQC.calculateTransitionMaxBoundaryIntensityNormalized(chrom_data, sum_transition=False) #sum (sum.light | sum.heavy)
    if transition is not None:
      return [
        # light transition
        TransitionJaggedness[transition + '.light'][0],
        TransitionSymmetry[transition + '.light'][0],
        TransitionModality[transition + '.light'][0],
        TransitionShift[transition + '.light'][0],
        TransitionFWHM[transition + '.light'][0],
        TransitionFWHM2base[transition + '.light'][0],
        TransitionMaxBoundaryIntensityNormalized[transition + '.light'][0],
        # heavy transition
        TransitionJaggedness[transition + '.heavy'][0],
        TransitionSymmetry[transition + '.heavy'][0],
        TransitionModality[transition + '.heavy'][0],
        TransitionShift[transition + '.heavy'][0],
        TransitionFWHM[transition + '.heavy'][0],
        TransitionFWHM2base[transition + '.heavy'][0],
        TransitionMaxBoundaryIntensityNormalized[transition + '.heavy'][0],
        #transition pair
        # PairRatioConsistency[transition][0],
        PairSimilarity[transition][0],
        PairShift[transition][0],
        PairFWHMConsistency[transition][0]
      ]
    else:
      transitions = chrom_data['transitions']
      a = pd.concat([TransitionJaggedness, TransitionSymmetry, TransitionModality, TransitionShift, TransitionFWHM, TransitionFWHM2base, TransitionMaxBoundaryIntensityNormalized])
      transition_light = a[map(lambda x: x + '.light', transitions)]
      transition_light.columns = transitions
      transition_heavy =  a[map(lambda x: x + '.heavy', transitions)]
      transition_heavy.columns = transitions
      transition_pair = pd.concat([PairSimilarity, PairShift, PairFWHMConsistency])
      return pd.concat([transition_light, transition_heavy, transition_pair]).T.values.tolist()

  def serializeLevel2(self, chrom = None, fn=None, ps=None, transitions=None):
    if chrom is None:
      chrom_data = self.selectChromData(fn, ps, transitions)
    else:
      chrom_data = chrom
    if chrom_data is None:
      return None
    PeakGroupRatioCorr = self.TargetedMSQC.calculatePeakGroupRatioCorr(chrom_data)
    PeakGroupJaggedness = self.TargetedMSQC.calculatePeakGroupJaggedness(chrom_data, flatness_factor=0.05)
    PeakGroupSymmetry = self.TargetedMSQC.calculatePeakGroupSymmetry(chrom_data)
    PeakGroupSimilarity = self.TargetedMSQC.calculatePeakGroupSimilarity(chrom_data)
    PeakGroupModality = self.TargetedMSQC.calculatePeakGroupModality(chrom_data, flatness_factor=0.05)
    PeakGroupShift = self.TargetedMSQC.calculatePeakGroupShift(chrom_data)
    PeakGroupFWHM = self.TargetedMSQC.calculatePeakGroupFWHM(chrom_data)
    PeakGroupFWHM2base = self.TargetedMSQC.calculatePeakGroupFWHM2base(chrom_data)

    IsotopeJaggedness = self.TargetedMSQC.calculateIsotopeJaggedness(chrom_data, flatness_factor=0.05) 
    IsotopeSymmetry = self.TargetedMSQC.calculateIsotopeSymmetry(chrom_data) 
    IsotopeSimilarity = self.TargetedMSQC.calculateIsotopeSimilarity(chrom_data) 
    IsotopeModality = self.TargetedMSQC.calculateIsotopeModality(chrom_data, flatness_factor=0.05) 
    IsotopeShift = self.TargetedMSQC.calculateIsotopeShift(chrom_data) 
    IsotopeFWHM = self.TargetedMSQC.calculateIsotopeFWHM(chrom_data)
    IsotopeFWHM2base = self.TargetedMSQC.calculateIsotopeFWHM2base(chrom_data)
    return [
      IsotopeJaggedness['light'][0],
      IsotopeSymmetry['light'][0],
      IsotopeSimilarity['light'][0],
      IsotopeModality['light'][0],
      IsotopeShift['light'][0],
      IsotopeFWHM['light'][0],
      IsotopeFWHM2base ['light'][0],

      IsotopeJaggedness['heavy'][0],
      IsotopeSymmetry['heavy'][0],
      IsotopeSimilarity['heavy'][0],
      IsotopeModality['heavy'][0],
      IsotopeShift['heavy'][0],
      IsotopeFWHM['heavy'][0],
      IsotopeFWHM2base ['heavy'][0],

      PeakGroupRatioCorr,
      PeakGroupJaggedness,
      PeakGroupSymmetry,
      PeakGroupSimilarity,
      PeakGroupModality,
      PeakGroupShift,
      PeakGroupFWHM,
      PeakGroupFWHM2base
    ]

  def serializeLevel3(self, ps=None, show_labels=False, chrom_data_list=[]):
    if len(chrom_data_list) == 0:
      chrom_data_list = []
      for fn in self.filename_list:
        chrom_data_list.append(self.selectChromData(fileName=fn, pepModSeq=ps))
    PairRatioConsistency = self.TargetedMSQC.calculatePairRatioConsistency(chrom_data_list)
    MeanIsotopeRatioConsistency = self.TargetedMSQC.calculateMeanIsotopeRatioConsistency(chrom_data_list)
    MeanIsotopeFWHMConsistency = self.TargetedMSQC.calculateMeanIsotopeFWHMConsistency(chrom_data_list)
    Area2SumRatioCV = self.TargetedMSQC.calculateArea2SumRatioCV(chrom_data_list)
    MeanIsotopeRTConsistency = self.TargetedMSQC.calculateMeanIsotopeRTConsistency(chrom_data_list)
    r = re.compile(".*light")
    lightTransitions = list(filter(r.match, MeanIsotopeRatioConsistency.columns))
    r = re.compile(".*heavy")
    heavyTransitions= list(filter(r.match, MeanIsotopeRatioConsistency.columns))

    if len(lightTransitions) == 0 or len(heavyTransitions) == 0:
      if show_labels:
        return pd.DataFrame([], columns = ["MeanIsotopeRatioConsistency_light", "MeanIsotopeFWHMConsistency_light", "Area2SumRatioCV_light",  "MeanIsotopeRatioConsistency_heavy", "MeanIsotopeFWHMConsistency_heavy", "Area2SumRatioCV_heavy", "MeanIsotopeRTConsistency", "PairRatioConsistency"])
      else:
        return []
    
    MeanIsotopeRatioConsistency_light = pd.melt(MeanIsotopeRatioConsistency[lightTransitions], ignore_index=False)
    MeanIsotopeRatioConsistency_heavy = pd.melt(MeanIsotopeRatioConsistency[heavyTransitions], ignore_index=False)
    
    MeanIsotopeFWHMConsistency_light = pd.melt(MeanIsotopeFWHMConsistency[lightTransitions], ignore_index=False)
    MeanIsotopeFWHMConsistency_heavy = pd.melt(MeanIsotopeFWHMConsistency[heavyTransitions], ignore_index=False)

    Area2SumRatioCV_light = Area2SumRatioCV[lightTransitions].T
    Area2SumRatioCV_light['transition'] = list(map(lambda x: '.'.join(x.split('.')[0:-1]), Area2SumRatioCV_light.index))
    Area2SumRatioCV_heavy = Area2SumRatioCV[heavyTransitions].T
    Area2SumRatioCV_heavy['transition'] = list(map(lambda x: '.'.join(x.split('.')[0:-1]), Area2SumRatioCV_heavy.index))
    
    transitions = pd.DataFrame(map(lambda x: '.'.join(x.split('.')[0:-1]), MeanIsotopeRatioConsistency_light['variable']))
    transitions.index = MeanIsotopeRatioConsistency_light.index
    combined = pd.concat([transitions, MeanIsotopeRatioConsistency_light['value'], MeanIsotopeRatioConsistency_heavy['value'], MeanIsotopeFWHMConsistency_light['value'], MeanIsotopeFWHMConsistency_heavy['value']], axis=1)
    combined.columns = ['transition', 'MeanIsotopeRatioConsistency_light', 'MeanIsotopeRatioConsistency_heavy', 'MeanIsotopeFWHMConsistency_light', 'MeanIsotopeFWHMConsistency_heavy']


    combined = combined.merge(Area2SumRatioCV_light, left_on="transition", right_on="transition", how="left")
    combined = combined.merge(Area2SumRatioCV_heavy, left_on="transition", right_on="transition", how="left")
    combined.index = transitions.index
    combined = combined.merge(MeanIsotopeRTConsistency, left_index=True, right_index=True)
    combined = combined.rename(columns={combined.columns[5]: 'Area2SumRatioCV_light', combined.columns[6]: 'Area2SumRatioCV_heavy', combined.columns[7]: 'MeanIsotopeRTConsistency'})

    PairRatioConsistencyList = []
    for (index, row) in combined.iterrows():
      transition = row['transition']
      PairRatioConsistencyList.append(PairRatioConsistency[transition][index])
    combined = combined.assign(PairRatioConsistency = PairRatioConsistencyList)
    combined = combined.dropna()
    if show_labels:
      return combined
    else:
      combined = combined[["MeanIsotopeRatioConsistency_light", "MeanIsotopeFWHMConsistency_light", "Area2SumRatioCV_light",  "MeanIsotopeRatioConsistency_heavy", "MeanIsotopeFWHMConsistency_heavy", "Area2SumRatioCV_heavy", "MeanIsotopeRTConsistency", "PairRatioConsistency"]]
      return combined.values.tolist()

  def serializeLevel1Title(self):
    return [
      'TransitionJaggedness_light',
      'TransitionSymmetry_light',
      'TransitionModality_light',
      'TransitionShift_light',
      'TransitionFWHM_light',
      'TransitionFWHM2base_light',
      'TransitionMaxBoundaryIntensityNormalized_light',
      'TransitionJaggedness_heavy',
      'TransitionSymmetry_heavy',
      'TransitionModality_heavy',
      'TransitionShift_heavy',
      'TransitionFWHM_heavy',
      'TransitionFWHM2base_heavy',
      'TransitionMaxBoundaryIntensityNormalized_heavy',
      'PairSimilarity',
      'PairShift',
      'PairFWHMConsistency'
    ]
  
  def serializeLevel2Title(self):
    return [
      'IsotopeJaggedness_light',
      'IsotopeSymmetry_light',
      'IsotopeSimilarity_light',
      'IsotopeModality_light',
      'IsotopeShift_light',
      'IsotopeFWHM_light',
      'IsotopeFWHM2base_light',
      'IsotopeJaggedness_heavy',
      'IsotopeSymmetry_heavy',
      'IsotopeSimilarity_heavy',
      'IsotopeModality_heavy',
      'IsotopeShift_heavy',
      'IsotopeFWHM_heavy',
      'IsotopeFWHM2base_heavy',
      'PeakGroupRatioCorr',
      'PeakGroupJaggedness',
      'PeakGroupSymmetry',
      'PeakGroupSimilarity',
      'PeakGroupModality',
      'PeakGroupShift',
      'PeakGroupFWHM',
      'PeakGroupFWHM2base'
    ]
  def serializeLevel3Title(self):
    return [
      "MeanIsotopeRatioConsistency_light",
      "MeanIsotopeFWHMConsistency_light",
      "Area2SumRatioCV_light",
      "MeanIsotopeRatioConsistency_heavy",
      "MeanIsotopeFWHMConsistency_heavy",
      "Area2SumRatioCV_heavy",
      "MeanIsotopeRTConsistency",
      "PairRatioConsistency"
    ]
