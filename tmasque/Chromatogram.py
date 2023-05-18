import time
import re
import pandas as pd
import numpy as np

class Chromatogram():
  def __init__(self,  chromatogramFile, boundaryFile=None, switchLightHeavy=False):
    start_time = time.time()
    self.chrom_path = chromatogramFile
    self.boundary_path = boundaryFile
    self.chrom = pd.read_csv(chromatogramFile, sep='\t')
    self.chrom.columns = self.chrom.columns.str.replace(' ', '')
    self.chrom['IsotopeLabelType'] = self.chrom['IsotopeLabelType'].str.lower()
    self.switchLightHeavy = switchLightHeavy
    chromatogram_data_load_time = time.time()
    print('Chromatogram data loaded in %.2f seconds' % (chromatogram_data_load_time - start_time))
    self.peak_boundary = None
    if boundaryFile is not None:
      self.peak_boundary = pd.read_csv(boundaryFile)
      self.peak_boundary.columns = self.peak_boundary.columns.str.replace(' ', '')
    peak_boundary_load_time = time.time()
    print('Peak boundary data loaded in %.2f seconds' % (peak_boundary_load_time - chromatogram_data_load_time))
    if switchLightHeavy:
      light_idx = self.chrom[self.chrom['IsotopeLabelType'] == 'light'].index.tolist()
      heavy_idx = self.chrom[self.chrom['IsotopeLabelType'] == 'heavy'].index.tolist()
      self.chrom.loc[light_idx, ['IsotopeLabelType']] = 'heavy'
      self.chrom.loc[heavy_idx, ['IsotopeLabelType']] = 'light'

  def getChromRTInst(self, fileName, pepModSeq, transitions=None):
    sample = self.chrom[(self.chrom['FileName'] == fileName) & (self.chrom['PeptideModifiedSequence'] == pepModSeq)]
    if transitions is not None:
      sample = sample[(sample['PrecursorCharge'].astype(str) + '.' + sample['FragmentIon'] + '.' +  sample['ProductCharge'].astype(str)).isin(transitions)]
    sample = sample.sort_values(by=['IsotopeLabelType','FragmentIon'], ascending=[False,True])
    max_len = 0
    for eachTime in list(sample['Times']):
      a = np.array(list(map(float,eachTime.split(','))))
      if len(a) > max_len:
        max_len = len(a)

    all_time = np.array([], dtype='float')
    for i in list(sample['Times']):
      a = np.array(list(map(float,i.split(','))))
      if len(a) == max_len:
        all_time = np.array(a, dtype='float')
        break
        
    total_col_name = ['.'.join(i) for i in list(zip(map(str, sample['PrecursorCharge']), sample['FragmentIon'], map(str,sample['ProductCharge']), sample['IsotopeLabelType']))]
    intensity = []
    for i in list(sample['Intensities']):
      a = np.array(list(map(float,i.split(','))))
      if len(a) < max_len:
        diff = max_len - len(a)
        a = np.pad(a, (diff, 0), 'constant', constant_values=0)
      intensity.append(a)
    intensity = pd.DataFrame(np.matrix(np.array(intensity, dtype='float')).T)
    intensity.columns = total_col_name
    return all_time, intensity
    
  
  def getChromData(self, fileName, pepModSeq, start=None, end=None, transitions=None):
    all_time, intensity = self.getChromRTInst(fileName, pepModSeq, transitions)
    if self.peak_boundary is None:
      if start is None or end is None:
        done_draw_rt = False
        while not done_draw_rt:
          draw_rts = np.random.choice(all_time, 2, replace=False)
          draw_rts.sort()
          start = draw_rts[0]
          end = draw_rts[1]
          if end - start >= 0.02:
            done_draw_rt = True
    else:
      if start is None:
        start = list(self.peak_boundary[(self.peak_boundary['FileName']==fileName)&(self.peak_boundary['PeptideModifiedSequence']==pepModSeq)]['MinStartTime'])[0]
      if end is None:
        end = list(self.peak_boundary[(self.peak_boundary['FileName']==fileName)&(self.peak_boundary['PeptideModifiedSequence']==pepModSeq)]['MaxEndTime'])[0]
      try:
        start = float(start)
        end = float(end)
        if np.isnan(start) or np.isnan(end):
          return None
      except:
        return None
    if start > end:
      temp = end
      end = start
      start = temp
    return self.make_chrom_obj(fileName, pepModSeq, all_time, intensity, start, end)
  
  def make_chrom_without_boundary(self, fileName, pepModSeq, rt, intensity):
    r = re.compile(".*light")
    endogenous_cols = np.unique(list(filter(r.match, intensity.columns))).tolist()
    r = re.compile(".*heavy")
    standard_cols = np.unique(list(filter(r.match,intensity.columns))).tolist()
    if len(standard_cols) == 0:
      return None
    light_transitions = list(map(lambda x: '.'.join(x.split('.')[0:-1]), endogenous_cols))
    heavy_transitions = list(map(lambda x: '.'.join(x.split('.')[0:-1]), standard_cols))
    # interset_set = set(light_transitions) & set(heavy_transitions)
    intersect_transitions = np.unique([x for x in heavy_transitions if x in (set(light_transitions) & set(heavy_transitions))]).tolist()
    endogenous_cols = list(map(lambda x: x + '.light', intersect_transitions))
    standard_cols = list(map(lambda x: x + '.heavy', intersect_transitions))
    total_col_name = endogenous_cols + standard_cols
    intensity = intensity[total_col_name]
    return dict(
      fileName = fileName,
      peptideModifiedSequence = pepModSeq,
      start = None, #left boundary RT
      end = None, #right boundary RT
      peak_time = None, #peak signal RT
      peak_intensity = None, #peak signal (stripped)
      peak_sum_intensity = None, #sum of same isotope(heavy, light) peak signal
      Area2SumRatio = None, #AUC/sum(all transition's AUC) of peak signal for each transition
      area = None,
      endogenous_cols = endogenous_cols,
      standard_cols = standard_cols,
      transitions = intersect_transitions,
      intensity= intensity,
      time = rt
    )
  
  def make_chrom_obj(self, fileName, pepModSeq, rt, intensity, start, end):
    peak_filter = (rt >= start) & (rt <= end)
    peak_time = rt[peak_filter]
    peak_intensity = intensity[peak_filter].reset_index(drop=True)
    #get column name for endogenous & sntandard
    r = re.compile(".*light")
    endogenous_cols = np.unique(list(filter(r.match, intensity.columns))).tolist()
    r = re.compile(".*heavy")
    standard_cols = np.unique(list(filter(r.match,intensity.columns))).tolist()
    if len(standard_cols) == 0:
      return None
    light_transitions = list(map(lambda x: '.'.join(x.split('.')[0:-1]), endogenous_cols))
    heavy_transitions = list(map(lambda x: '.'.join(x.split('.')[0:-1]), standard_cols))
    # interset_set = set(light_transitions) & set(heavy_transitions)
    intersect_transitions = np.unique([x for x in heavy_transitions if x in (set(light_transitions) & set(heavy_transitions))]).tolist()
    # TODO: some IsotopeLabelType has values in addition to light and heavy. (e.g. 15N heavy)

    endogenous_cols = list(map(lambda x: x + '.light', intersect_transitions))
    standard_cols = list(map(lambda x: x + '.heavy', intersect_transitions))
    
    total_col_name = endogenous_cols + standard_cols
    peak_intensity = peak_intensity[total_col_name]
    intensity = intensity[total_col_name]
    sum_columns = ['sum.light','sum.heavy']
    peak_sum_intensity = pd.DataFrame(zip(peak_intensity[endogenous_cols].sum(1),peak_intensity[standard_cols].sum(1)), columns=sum_columns)
    
    #3. Area2SumRatio (Note: this value is calculated for all transitions)
    area = []
    for c in total_col_name:
      area.append(np.trapz(peak_intensity[c], peak_time))
    area = pd.DataFrame(area).transpose()
    area.columns = peak_intensity.columns
    
    Area2SumRatio_endo = area[endogenous_cols]/sum(area[endogenous_cols].loc[0])
    Area2SumRatio_stand = area[standard_cols]/sum(area[standard_cols].loc[0])
    Area2SumRatio = pd.concat([Area2SumRatio_endo, Area2SumRatio_stand], axis=1)
    return dict(
      fileName = fileName,
      peptideModifiedSequence = pepModSeq,
      start = start, #left boundary RT
      end = end, #right boundary RT
      peak_time = peak_time, #peak signal RT
      peak_intensity = peak_intensity, #peak signal (stripped)
      peak_sum_intensity = peak_sum_intensity, #sum of same isotope(heavy, light) peak signal
      Area2SumRatio = Area2SumRatio, #AUC/sum(all transition's AUC) of peak signal for each transition
      area = area,
      endogenous_cols = endogenous_cols,
      standard_cols = standard_cols,
      transitions = intersect_transitions,
      intensity= intensity,
      time = rt
    )
