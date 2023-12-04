import statistics
import re
import os
import sys
import pandas as pd
import numpy as np
import math
from .PeakQualityControl import PeakQualityControl
from .QualityEncoder import QualityEncoder
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def _get_peak_top_and_area(params):
  peptide = params['peptide']
  file = params['file']
  qc = params['qc']
  chrom = qc.selectChromData(file, peptide)
  if chrom is None:
    return (peptide, file, None, None, None)
  max_peak_intensity = chrom['peak_intensity'].max()
  transitions = chrom['transitions']
  peak_area = chrom['area']
  return (peptide, file, transitions, max_peak_intensity, peak_area)

def _chrom_plot_page(options):
  page_items = options['page_items']
  score_tables = options['score_tables']
  nrow = options['nrow']
  ncol = options['ncol']
  plt.rcParams['figure.dpi'] = options['dpi']
  plt.rcParams['figure.figsize'] = (options['fig_w'], options['fig_h'])
  plt.rcParams.update({'font.size': 10})
  switchLightHeavy = True if options['switchLightHeavy'] else False
  fig, axs = plt.subplots(nrow*2, ncol, layout="constrained")
  item_idx = 0
  if page_items[0] is not None:
    if options['mol_separation']:
      fig.suptitle(page_items[0]['peptideModifiedSequence'] + ' (Page ' + str(options['page_num']) + ' of ' + str(options['total_page']) + ')', fontsize=14, fontweight='bold', ha='left', x=0.01)
    else:
      fig.suptitle('Page ' + str(options['page_num']) + ' of ' + str(options['total_page']), fontsize=10, ha='right', x=0.98)
#   fig.title(, fontsize=10, loc='right')
  for row_idx in range(nrow):
    for col_idx in range(ncol):
      if item_idx >= len(page_items) or page_items[item_idx] is None:
        axs[2*row_idx, col_idx].remove()
        axs[2*row_idx + 1, col_idx].remove()
        continue
      chrom_data = page_items[item_idx]
      fn = chrom_data['fileName']
      ps = chrom_data['peptideModifiedSequence']
      #score_table = pd.DataFrame(self.cache_store[ps][fn]).T
      try:
        score_table = score_tables[ps][fn]
      except:
        score_table = None
      _sub_chrom_plot(chrom_data, score_table, axs[2*row_idx, col_idx], axs[2*row_idx + 1, col_idx], options = options, mol_separation = options['mol_separation'], switchLightHeavy=switchLightHeavy)
      item_idx += 1
  return fig

def _sub_chrom_plot(chrom_data, store_table, ax1, ax2, options=None, mol_separation=True, switchLightHeavy=False):
  fn = chrom_data['fileName']
  ps = chrom_data['peptideModifiedSequence']
  time = chrom_data['time']
  light_global_max_intensity = chrom_data['intensity'][chrom_data['endogenous_cols']].max().max()
  heavy_global_max_intensity = chrom_data['intensity'][chrom_data['standard_cols']].max().max()
  light_exp = len(str(int(light_global_max_intensity))) - 2
  heavy_exp = len(str(int(heavy_global_max_intensity))) - 2
  if store_table is not None:
    type2_score = store_table['type2_score'][0]
    type3_score = store_table['type3_score'].median()
    type1_score = store_table['type1_score'].median()
  else:
    type1_score = None
    type2_score = None
    type3_score = None
  if options is None or options['type2_min'] is None or options['type2_max'] is None: 
    type2_max = 7.2
    type2_min = 5.0
  else:
    type2_max = options['type2_max']
    type2_min = options['type2_min']

  if options is None or options['type1_min'] is None or options['type1_max'] is None:
    type1_min = 6.7
    type1_max = 7.7
  else:
    type1_min = options['type1_min']
    type1_max = options['type1_max']

  if options is None or options['type3_min'] is None or options['type3_max'] is None:
    type3_min = 7.901
    type3_max = 8.665
  else:
    type3_min = options['type3_min']
    type3_max = options['type3_max']

  if light_exp < 0 :
      light_exp = 0
  if heavy_exp < 0:
      heavy_exp = 0
  for fragment in chrom_data['transitions']:
      if switchLightHeavy:
        ax1.plot(time, [intensity/(10**light_exp) for intensity in chrom_data['intensity'][fragment + '.light']],  linewidth=1, label=fragment + '.heavy')
        ax2.plot(time, [intensity/(10**heavy_exp) for intensity in chrom_data['intensity'][fragment + '.heavy']],  linewidth=1, label=fragment + '.light')
      else:
        ax1.plot(time, [intensity/(10**light_exp) for intensity in chrom_data['intensity'][fragment + '.light']],  linewidth=1, label=fragment + '.light')
        ax2.plot(time, [intensity/(10**heavy_exp) for intensity in chrom_data['intensity'][fragment + '.heavy']],  linewidth=1, label=fragment + '.heavy')
      if not mol_separation:
        ax1.set_title(ps + '\n', fontsize=12, fontweight='bold')
      ax1.annotate(fn, xy=(0.5, 1.02), xycoords='axes fraction', ha='center', va='bottom', fontsize=10, fontweight='normal')
      ax1.set_ylabel('Intensity(10^%s)'%light_exp, fontsize=10)
      ax1.set_xlabel('Retention Time', fontsize = 10)
      if chrom_data['start'] is not None:
        ax1.axvline(chrom_data['start'], linestyle= '--', linewidth=1,color='black')
      if chrom_data['end'] is not None:
        ax1.axvline(chrom_data['end'], linestyle= '--', linewidth=1,color='black')
      ax1.legend(fontsize=7, loc='best', frameon=False)
#       ax1.annotate('Type 1: ' + str(round(type1_score, 2)), fontsize=8, xy=(0, 1), xycoords='axes fraction',  ha='left', va='top')
#         ax1.annotate(fn, fontsize=6, xy=(0, 1.05), xycoords="axes fraction")
#         ax1.legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1), frameon=False)\
      ax2.set_ylabel('Intensity(10^%s)'%heavy_exp, fontsize = 10)
      ax2.xaxis.tick_top()
      if chrom_data['start'] is not None:
        ax2.axvline(chrom_data['start'], linestyle= '--', linewidth=1,color='black')
      if chrom_data['end'] is not None:
        ax2.axvline(chrom_data['end'], linestyle= '--', linewidth=1,color='black')
      ax2.legend(fontsize=7, loc='best', frameon=False)
      
#       good_format = workbook.add_format({'bg_color': '#00FF00'}),
#       warn_format = workbook.add_format({'bg_color': '#ffc107'}),
#       bad_format = workbook.add_format({'bg_color': 'red'})
  good_color = '#198754'
  warn_color = '#cc9a06' 
  bad_color = '#FF0000'
  if type1_score is not None:
    if type1_score > type1_max:
      type1_color = good_color
      type1_quality = 'Good'
    elif type1_score < type1_min:
      type1_color = bad_color
      type1_quality = 'Poor'
    else:
      type1_color = warn_color
      type1_quality = 'Acceptable'
  else:
    type1_color = 'black'
    type1_quality = 'N/A'
  
  if type2_score is not None:
    if type2_score > type2_max:
      type2_color = good_color
      type2_quality = 'Good'
    elif type2_score < type2_min:
      type2_color = bad_color
      type2_quality = 'Poor'
    else:
      type2_color = warn_color
      type2_quality = 'Acceptable'
  else:
    type2_color = 'black'
    type2_quality = 'N/A'

  
  if type3_score is not None:
    if type3_score > type3_max:
      type3_color = good_color
      type3_quality = 'Good'
    elif type3_score < type3_min:
      type3_color = bad_color
      type3_quality = 'Poor'
    else:
      type3_color = warn_color
      type3_quality = 'Acceptable'
  else:
    type3_color = 'black'
    type3_quality = 'N/A'
  if type1_score is not None:
    ax2.annotate('Type 1: ' + str(round(type1_score, 2)) + '\n' + type1_quality + '\n\n', xy=(-0.05, -0.03), xycoords='axes fraction', ha='left', va='top', fontweight="bold", fontsize=10, color=type1_color)
  else:
    ax2.annotate('Type 1: N/A\n \n\n', xy=(-0.05, -0.03), xycoords='axes fraction', ha='left', va='top', fontweight="bold", fontsize=10, color=type1_color)
  if type2_score is not None:
    ax2.annotate('Type 2: ' + str(round(type2_score, 2)) + '\n' + type2_quality + '\n\n', xy=(0.5, -0.03), xycoords='axes fraction', ha='center', va='top', fontweight="bold", fontsize=10, color=type2_color)
  else:
    ax2.annotate('Type 2: N/A\n \n\n', xy=(0.5, -0.03), xycoords='axes fraction', ha='center', va='top', fontweight="bold", fontsize=10, color=type2_color)
  if type3_score is not None:
    ax2.annotate('Type 3: ' + str(round(type3_score, 2)) + '\n' + type3_quality + '\n\n', xy=(1.05, -0.03), xycoords='axes fraction', ha='right', va='top', fontweight="bold", fontsize=10, color=type3_color)
  else:
    ax2.annotate('Type 3: N/A\n \n\n', xy=(1.05, -0.03), xycoords='axes fraction', ha='right', va='top', fontweight="bold", fontsize=10, color=type3_color)
  
  if type2_score is not None:
    if type2_score >= type2_max:
        ax1.set_facecolor('mintcream')
        ax2.set_facecolor('mintcream')
    elif type2_score < type2_max and type2_score >= type2_min:
        ax1.set_facecolor('cornsilk')
        ax2.set_facecolor('cornsilk')
    elif type2_score < type2_min:
        ax1.set_facecolor('mistyrose')
        ax2.set_facecolor('mistyrose')
  else:
    ax1.set_facecolor('lightgray')
    ax2.set_facecolor('lightgray')

class TargetedMSQualityEncoder():
  def __init__(self, chromatogram_tsv, peak_boundary_csv, core_num=20, switchLightHeavy=False, device='auto'):
    self.core_num = core_num
    self.peak_qc = PeakQualityControl(chromatogram_tsv, peak_boundary_csv, core_num=self.core_num, switchLightHeavy=switchLightHeavy)
    self.device = device
    self.cache_store = {}
    self._sample_df = None
    self._transition_df = None
    self.no_boundary_set = dict()
    self.switchLightHeavy = switchLightHeavy
    self.quality_warn_range_config = dict(
      # Type 1 quality
      TransitionJaggedness_light = dict(min=0, max=0.3, direction=-1),
      TransitionSymmetry_light=dict(min=0, max=1, direction=1),
      TransitionModality_light=dict(min=0, max=0.3, direction=-1),
      TransitionShift_light=dict(min=0, max=0.3, direction=-1),
      TransitionFWHM_light=dict(min=0.05, max=1, direction=-1),
      TransitionFWHM2base_light=dict(min=0.25, max=1, direction=-1),
      TransitionMaxBoundaryIntensityNormalized_light=dict(min=0, max=1, direction=-1),
      
      TransitionJaggedness_heavy = dict(min=0, max=0.3, direction=-1),
      TransitionSymmetry_heavy=dict(min=0, max=1, direction=1),
      TransitionModality_heavy=dict(min=0, max=0.3, direction=-1),
      TransitionShift_heavy=dict(min=0, max=0.3, direction=-1),
      TransitionFWHM_heavy=dict(min=0.05, max=1, direction=-1),
      TransitionFWHM2base_heavy=dict(min=0.25, max=1, direction=-1),
      TransitionMaxBoundaryIntensityNormalized_heavy=dict(min=0, max=1, direction=-1),

      PairSimilarity=dict(min=0.7, max=1, direction=1),
      PairShift=dict(min=0, max=0.3, direction=-1),
      PairFWHMConsistency=dict(min=0, max=1, direction=-1),

      # Type 2 quality
      IsotopeJaggedness_light=dict(min=0, max=0.3, direction=-1),
      IsotopeSymmetry_light=dict(min=0, max=1, direction=1),
      IsotopeSimilarity_light=dict(min=0.7, max=1, direction=1),
      IsotopeModality_light=dict(min=0, max=0.3, direction=-1),
      IsotopeShift_light=dict(min=0, max=0.3, direction=-1),
      IsotopeFWHM_light=dict(min=0.2, max=1, direction=-1),
      IsotopeFWHM2base_light=dict(min=0.2, max=1, direction=-1),
      
      IsotopeJaggedness_heavy=dict(min=0, max=0.3, direction=-1),
      IsotopeSymmetry_heavy=dict(min=0, max=1, direction=1),
      IsotopeSimilarity_heavy=dict(min=0.7, max=1, direction=1),
      IsotopeModality_heavy=dict(min=0, max=0.3, direction=-1),
      IsotopeShift_heavy=dict(min=0, max=0.3, direction=-1),
      IsotopeFWHM_heavy=dict(min=0.2, max=1, direction=-1),
      IsotopeFWHM2base_heavy=dict(min=0.2, max=1, direction=-1),

      PeakGroupRatioCorr=dict(min=0.7, max=1, direction=1),
      PeakGroupJaggedness=dict(min=0, max=0.3, direction=-1),
      PeakGroupSymmetry=dict(min=0, max=1, direction=1),
      PeakGroupSimilarity=dict(min=0.7, max=1, direction=1),
      PeakGroupModality=dict(min=0, max=0.3, direction=-1),
      PeakGroupShift=dict(min=0, max=0.3, direction=-1),
      PeakGroupFWHM=dict(min=0.2, max=1, direction=-1),
      PeakGroupFWHM2base=dict(min=0.2, max=1, direction=-1),

      #Type 3 quality
      MeanIsotopeRatioConsistency_light=dict(min=0, max=0.5, direction=-1),
      MeanIsotopeFWHMConsistency_light=dict(min=0, max=1, direction=-1),
      Area2SumRatioCV_light=dict(min=0, max=0.5, direction=-1),
      MeanIsotopeRatioConsistency_heavy=dict(min=0, max=0.5, direction=-1),
      MeanIsotopeFWHMConsistency_heavy=dict(min=0, max=1, direction=-1),
      Area2SumRatioCV_heavy=dict(min=0, max=0.5, direction=-1),
      MeanIsotopeRTConsistency=dict(min=0, max=0.5, direction=-1),
      PairRatioConsistency=dict(min=0, max=0.5, direction=-1)
    )
    self.identify_no_boundary_set()
  def identify_no_boundary_set(self):
    # Identify samples without min start time and max end time
    pb = self.peak_qc.ms_data.peak_boundary
    no_pb = pb[(pb['MinStartTime'].isna()) | (pb['MaxEndTime'].isna())]
    self.no_boundary_set = dict()
    for _, row in no_pb.iterrows():
      ps = row['PeptideModifiedSequence']
      fn = row['FileName']
      if ps not in self.no_boundary_set:
        self.no_boundary_set[ps] = set()
      self.no_boundary_set[ps].add(fn)

  def run_encoder(self, encoder_type1_path=None, encoder_type2_path=None, encoder_type3_path=None, encoder_dim=[2, 2, 2]):
    self.quality_encoder = QualityEncoder(device=self.device, encoder_type1_path=encoder_type1_path, encoder_type2_path=encoder_type2_path, encoder_type3_path=encoder_type3_path, encoder_dim=encoder_dim)
    self.qc_values = [
      self.peak_qc.calculateQualityType1(),
      self.peak_qc.calculateQualityType2(),
      self.peak_qc.calculateQualityType3()
    ]
    self.qc_scores = [
      self.quality_encoder(list(map(lambda x: x['data'], self.qc_values[0]))),
      self.quality_encoder(list(map(lambda x: x['data'], self.qc_values[1]))),
      self.quality_encoder(list(map(lambda x: x['data'], self.qc_values[2])))
    ]
    print('Organizing quality information ...')
    self._organize_dataset()
    print('Calculating peaktop and peak area ...')
    self._append_peak_top_and_area()
    print('Quality Calculated')

  def summarize_dataset(self, file_group_delimiter=None, file_group_suffix_type=None):
    store = self.cache_store
    summarized = dict()
    outputData = dict()
    keysForTransitions = ['type1_score', 'type2_score', 'type3_score', 'light_peak_top', 'heavy_peak_top', 'light_peak_area', 'heavy_peak_area'] + self.peak_qc.serializeLevel1Title() + self.peak_qc.serializeLevel2Title() + self.peak_qc.serializeLevel3Title()
    keysForSamples = ['type1_score_median', 'type2_score', 'type3_score_median',
        'light_peak_top_max', 'heavy_peak_top_max', 'light_peak_area_max', 'heavy_peak_area_max', 
        'type1_score_max_ion', 'type3_score_max_ion', 'light_peak_top_max_ion', 'heavy_peak_top_max_ion', 'light_peak_area_max_ion', 'heavy_peak_area_max_ion']
    for peptide in store.keys():
      outputData[peptide] = dict()
      if peptide not in summarized:
          summarized[peptide] = dict()
      for sample in store[peptide].keys():
        outputData[peptide][sample] = dict()
        if sample not in summarized[peptide]:
            summarized[peptide][sample] = dict()
        transitions = store[peptide][sample].keys()
        for transition in transitions:
            keys = store[peptide][sample][transition].keys()
            for key in keys:
                if key not in summarized[peptide][sample]:
                    summarized[peptide][sample][key] = []
                summarized[peptide][sample][key].append(store[peptide][sample][transition][key])
        if file_group_delimiter is not None and file_group_suffix_type is not None:
          matched = re.search('(.*)' + file_group_delimiter + '(\\' + file_group_suffix_type + '+)\.(.*)$', sample)
          out = dict(peptide=peptide, sample=sample, basename=matched.group(1), group=str(matched.group(2)))
        else:
          out = dict(peptide=peptide, sample=sample)
        for key in summarized[peptide][sample].keys():
          if not re.match(r"(.*)_latent$", key) and not re.match(r"(.*)_value$", key):
            #out[key+ '_x'] = summarized[peptide][sample][key][0]
            #out[key+ '_y'] = summarized[peptide][sample][key][1]
            if re.match(r"(.*)type2_score", key):
                out['type2_score'] = summarized[peptide][sample][key][0]
            else:
              if re.match(r"(.*)type", key):
                max_index = np.argmax(summarized[peptide][sample][key])
                max_frag = list(store[peptide][sample].keys())[max_index]
                out[key+'_max_ion'] = max_frag
                out[key+ '_median'] = statistics.median(summarized[peptide][sample][key])
              else:
                max_index = np.argmax(summarized[peptide][sample][key])
                max_frag = list(store[peptide][sample].keys())[max_index]
                out[key+'_max_ion'] = max_frag
                out[key+ '_max'] = store[peptide][sample][max_frag][key]          
        outputData[peptide][sample] = out
    outputArr = []
    pb = self.peak_qc.ms_data.peak_boundary
    pb = pb[['PeptideModifiedSequence', 'FileName']].drop_duplicates()
    for idx, row in pb.iterrows():
      peptide = row['PeptideModifiedSequence']
      sample = row['FileName']
      if peptide in outputData and sample in outputData[peptide]:
        outputArr.append(outputData[peptide][sample])
      elif peptide in self.no_boundary_set and sample in self.no_boundary_set[peptide]:
        if file_group_delimiter is not None and file_group_suffix_type is not None:
          matched = re.search('(.*)' + file_group_delimiter + '(\\' + file_group_suffix_type + '+)\.(.*)$', sample)
          out = dict(peptide=peptide, sample=sample, basename=matched.group(1), group=str(matched.group(2)))
        else:
          out = dict(peptide=peptide, sample=sample)
        for key in keysForSamples:
          out[key] = None
        outputArr.append(out)
    outputTransitionData = dict()
    target2Transitions = dict()
    for peptide in store.keys():
      target2Transitions[peptide] = set()
      for sample in store[peptide].keys():
        transitions = store[peptide][sample].keys()
        if file_group_delimiter is not None and file_group_suffix_type is not None:
          matched = re.search('(.*)' + file_group_delimiter + '(\\' + file_group_suffix_type + '+)\.(.*)$', sample)
        for transition in transitions:
          target2Transitions[peptide].add(transition)
          if file_group_delimiter is not None and file_group_suffix_type is not None:
            out = dict(peptide=peptide, sample=sample, basename=matched.group(1), group=matched.group(2), transition=transition)
          else:
            out = dict(peptide=peptide, sample=sample, transition=transition)
          keys = store[peptide][sample][transition].keys()
          for key in keys:
            if not re.match(r"(.*)_latent$", key) and not re.match(r"(.*)_value$", key):
              out[key] = store[peptide][sample][transition][key]
            elif key == 'type1_value':
              type1_title = self.peak_qc.serializeLevel1Title()
              for idx, t in enumerate(type1_title):
                out[t] = store[peptide][sample][transition]['type1_value'][idx]
            elif key == 'type2_value':
              type2_title = self.peak_qc.serializeLevel2Title()
              for idx, t in enumerate(type2_title):
                out[t] = store[peptide][sample][transition]['type2_value'][idx]
            elif key == 'type3_value':
              type3_title = self.peak_qc.serializeLevel3Title()
              for idx, t in enumerate(type3_title):
                out[t] = store[peptide][sample][transition]['type3_value'][idx]
          if peptide not in outputTransitionData:
            outputTransitionData[peptide] = dict()
          if sample not in outputTransitionData[peptide]:
            outputTransitionData[peptide][sample] = dict()
          outputTransitionData[peptide][sample][transition] = out
    outputTransitionArr = []
    for idx, row in pb.iterrows():
      peptide = row['PeptideModifiedSequence']
      sample = row['FileName']
      if peptide in outputTransitionData and sample in outputTransitionData[peptide]:
        for transition in outputTransitionData[peptide][sample].keys():
          if outputTransitionData[peptide][sample][transition]:
            outputTransitionArr.append(outputTransitionData[peptide][sample][transition])
      elif peptide in self.no_boundary_set and sample in self.no_boundary_set[peptide]:
        transitions = target2Transitions[peptide]
        for transition in transitions:
          if file_group_delimiter is not None and file_group_suffix_type is not None:
            matched = re.search('(.*)' + file_group_delimiter + '(\\' + file_group_suffix_type + '+)\.(.*)$', sample)
            out = dict(peptide=peptide, sample=sample, basename=matched.group(1), group=matched.group(2), transition=transition)
          else:
            out = dict(peptide=peptide, sample=sample, transition=transition)
          for key in keysForTransitions:
            out[key]=None
          outputTransitionArr.append(out)
    self._sample_df = pd.DataFrame(outputArr)
    self._transition_df = pd.DataFrame(outputTransitionArr)
    if file_group_delimiter is not None and file_group_suffix_type is not None:
      self._sample_df = self._sample_df[['peptide', 'sample', 'basename', 'group'] + keysForSamples]
      self._transition_df = self._transition_df[['peptide', 'sample', 'basename', 'group', 'transition'] + keysForTransitions]
    else:
      self._sample_df = self._sample_df[['peptide', 'sample'] + keysForSamples]
      self._transition_df = self._transition_df[['peptide', 'sample', 'transition'] + keysForTransitions]
    return self._transition_df, self._sample_df
  def _get_workbook_formats(self, workbook, options=None):
    return dict(
      type1_color_format = {
        'type': '2_color_scale',
        'min_value': 6.7 if options is None or 'type1_min' not in options else options['type1_min'],
        'max_value': 7.7 if options is None or 'type1_max' not in options else options['type1_max'],
        'min_type': 'num',
        'max_type': 'num',
        'min_color': 'red',
        'max_color': '#00FF00'
      },
      type2_color_format = {
        'type': '2_color_scale',
        'min_value': 5.0 if options is None or 'type2_min' not in options else options['type2_min'],
        'max_value': 7.2 if options is None or 'type2_max' not in options else options['type2_max'],
        'min_type': 'num',
        'max_type': 'num',
        'min_color': 'red',
        'max_color': '#00FF00'
      },
      type3_color_format = {
        'type': '2_color_scale',
        'min_value': 7.901 if options is None or 'type3_min' not in options else options['type3_min'],
        'max_value': 8.665 if options is None or 'type3_max' not in options else options['type3_max'],
        'min_type': 'num',
        'max_type': 'num',
        'min_color': 'red',
        'max_color': '#00FF00'
      },
      peaktop_color_format = {
        'type': '2_color_scale',
        'min_value': 3 if options is None or 'peaktop_min' not in options else options['peaktop_min'],
        'max_value': 4 if options is None or 'peaktop_max' not in options else options['peaktop_max'],
        'min_type': 'num',
        'max_type': 'num',
        'min_color': 'red',
        'max_color': 'white'
      },
      peakarea_color_format = {
        'type': '2_color_scale',
        'min_value': 3 if options is None or 'peakarea_min' not in options else options['peakarea_min'],
        'max_value': 4 if options is None or 'peakarea_max' not in options else options['peakarea_max'],
        'min_type': 'num',
        'max_type': 'num',
        'min_color': 'red',
        'max_color': 'white'
      },
      title_format = workbook.add_format({'bold': True, 'font_color': 'white', 'bg_color': 'black'}),
      digit_format = workbook.add_format({'num_format': '0.000'}),
      text_center = workbook.add_format({'align': 'center'}),
      top_border = workbook.add_format({'top': 2, 'border_color': 'black'}),
      left_thin_border = workbook.add_format({'left': 2, 'border_color': 'black'}),
      left_thick_border = workbook.add_format({'left': 5, 'border_color': 'black'}),
      thick_left_top_border = workbook.add_format({'left': 5, 'top': 2, 'border_color': 'black'}),
      thin_left_top_border = workbook.add_format({'left': 2, 'top': 2, 'border_color': 'black'}),
      good_format = workbook.add_format({'bg_color': '#00FF00'}),
      warn_format = workbook.add_format({'bg_color': '#ffc107'}),
      bad_format = workbook.add_format({'bg_color': 'red'})
    )
  
  def output_transition_quality(self, transition_output_folder, file_group_delimiter=None, file_group_suffix_type=None, options=None):
    os.makedirs(transition_output_folder, exist_ok =True)
    self.summarize_dataset(file_group_delimiter=file_group_delimiter, file_group_suffix_type=file_group_suffix_type)
    transition_df = self._transition_df
    for pep in self._transition_df['peptide'].unique():
      prcessed_pep = re.sub(r'\/', '_', pep)
      file_path = os.path.join(transition_output_folder, prcessed_pep + '.xlsx')
      print('Output transition qualities for ' + pep + ' in ' + file_path)
      df = transition_df[transition_df['peptide'] == pep].reset_index(drop=True)
      self.output_each_transition(df, file_path, file_group_delimiter=file_group_delimiter, file_group_suffix_type=file_group_suffix_type, options=None)
  
  def output_each_transition(self, transition_df, transition_output, file_group_delimiter=None, file_group_suffix_type=None, options=None):
    # print('Output Transition Quality file: ' + transition_output)
    transition_df_row_num = transition_df.shape[0]
    transition_df.rename(columns = {'peptide':'molecule'}, inplace = True)
    # transition_output = slugify(transition_output)
    with pd.ExcelWriter(transition_output, engine='xlsxwriter') as writer:
      transition_df.to_excel(writer, sheet_name='Transition Quality', index=False)
      workbook = writer.book
      workbook_formats = self._get_workbook_formats(workbook, options)
      type1_color_format = workbook_formats['type1_color_format']
      type2_color_format = workbook_formats['type2_color_format']
      type3_color_format = workbook_formats['type3_color_format']
      peaktop_color_format = workbook_formats['peaktop_color_format']
      peakarea_color_format = workbook_formats['peakarea_color_format']
      title_format = workbook_formats['title_format']
      digit_format = workbook_formats['digit_format']
      text_center = workbook_formats['text_center']
      top_border = workbook_formats['top_border']
      left_thin_border = workbook_formats['left_thin_border']
      left_thick_border = workbook_formats['left_thick_border']
      thick_left_top_border = workbook_formats['thick_left_top_border']
      thin_left_top_border = workbook_formats['thin_left_top_border']
      good_format = workbook_formats['good_format']
      warn_format = workbook_formats['warn_format']
      bad_format = workbook_formats['bad_format']
      writer.sheets['Transition Quality'].set_column(1, 20, None, text_center)
      col_idx_shift = 2 if 'group' in transition_df else 0
      for idx, column in enumerate(transition_df):
        if idx >= 10 + col_idx_shift:
          writer.sheets['Transition Quality'].write(0, idx, column, title_format)
          writer.sheets['Transition Quality'].set_column(idx, idx, 3)
        elif idx >= 3 + col_idx_shift:
          writer.sheets['Transition Quality'].write(0, idx, column, title_format)
          writer.sheets['Transition Quality'].set_column(idx, idx, 8)
        else:
          writer.sheets['Transition Quality'].write(0, idx, column, title_format)
          column_len = max(transition_df[column].astype(str).map(len).max(), len(column))
          writer.sheets['Transition Quality'].set_column(idx, idx, column_len + 4)
      writer.sheets['Transition Quality'].set_column(3 + col_idx_shift, 3 + col_idx_shift, None, left_thick_border)
      writer.sheets['Transition Quality'].set_column(6 + col_idx_shift, 6 + col_idx_shift, None, left_thick_border)
      writer.sheets['Transition Quality'].set_column(8 + col_idx_shift, 8 + col_idx_shift, None, left_thin_border)
      writer.sheets['Transition Quality'].set_column(10 + col_idx_shift, 10 + col_idx_shift, 3, left_thick_border)
      writer.sheets['Transition Quality'].set_column(17 + col_idx_shift, 17 + col_idx_shift, 3, left_thin_border)
      writer.sheets['Transition Quality'].set_column(24 + col_idx_shift, 24 + col_idx_shift, 3, left_thin_border)
      writer.sheets['Transition Quality'].set_column(27 + col_idx_shift, 27 + col_idx_shift, 3, left_thick_border)
      writer.sheets['Transition Quality'].set_column(34 + col_idx_shift, 34 + col_idx_shift, 3, left_thin_border)
      writer.sheets['Transition Quality'].set_column(41 + col_idx_shift, 41 + col_idx_shift, 3, left_thin_border)
      writer.sheets['Transition Quality'].set_column(49 + col_idx_shift, 49 + col_idx_shift, 3, left_thick_border)
      writer.sheets['Transition Quality'].set_column(57 + col_idx_shift, 57 + col_idx_shift, 3, left_thick_border)
      temp_id = None
      for index, row in transition_df.iterrows():
        id = row['molecule'] + row['sample']
        if temp_id is None:
          temp_id = id
        elif temp_id != id:
          writer.sheets['Transition Quality'].set_row(index + 1, None, top_border)
          writer.sheets['Transition Quality'].conditional_format(index + 1, 3 + col_idx_shift, index+1, 3 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
          writer.sheets['Transition Quality'].conditional_format(index + 1, 6 + col_idx_shift, index+1, 6 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
          writer.sheets['Transition Quality'].conditional_format(index + 1, 8 + col_idx_shift, index+1, 8 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
          writer.sheets['Transition Quality'].conditional_format(index + 1, 10 + col_idx_shift, index+1, 10 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
          writer.sheets['Transition Quality'].conditional_format(index + 1, 17 + col_idx_shift, index+1, 17 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
          writer.sheets['Transition Quality'].conditional_format(index + 1, 24 + col_idx_shift, index+1, 24 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
          writer.sheets['Transition Quality'].conditional_format(index + 1, 27 + col_idx_shift, index+1, 27 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
          writer.sheets['Transition Quality'].conditional_format(index + 1, 34 + col_idx_shift, index+1, 34 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
          writer.sheets['Transition Quality'].conditional_format(index + 1, 41 + col_idx_shift, index+1, 41 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
          writer.sheets['Transition Quality'].conditional_format(index + 1, 49 + col_idx_shift, index+1, 49 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
          writer.sheets['Transition Quality'].conditional_format(index + 1, 57 + col_idx_shift, index+1, 57 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
          temp_id = id
          # writer.sheets['Transition Quality'].set_row(index + 1, None, top_border)
      writer.sheets['Transition Quality'].set_row(index + 2, None, top_border)
      writer.sheets['Transition Quality'].conditional_format(index + 2, 3 + col_idx_shift, index+ 2, 3 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
      writer.sheets['Transition Quality'].conditional_format(index + 2, 6 + col_idx_shift, index+ 2, 6 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
      writer.sheets['Transition Quality'].conditional_format(index + 2, 8 + col_idx_shift, index+ 2, 8 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
      writer.sheets['Transition Quality'].conditional_format(index + 2, 10 + col_idx_shift, index+2, 10 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
      writer.sheets['Transition Quality'].conditional_format(index + 2, 17 + col_idx_shift, index+2, 17 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
      writer.sheets['Transition Quality'].conditional_format(index + 2, 24 + col_idx_shift, index+2, 24 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
      writer.sheets['Transition Quality'].conditional_format(index + 2, 27 + col_idx_shift, index+2, 27 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
      writer.sheets['Transition Quality'].conditional_format(index + 2, 34 + col_idx_shift, index+2, 34 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
      writer.sheets['Transition Quality'].conditional_format(index + 2, 41 + col_idx_shift, index+2, 41 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
      writer.sheets['Transition Quality'].conditional_format(index + 2, 49 + col_idx_shift, index+2, 49 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
      writer.sheets['Transition Quality'].conditional_format(index + 2, 57 + col_idx_shift, index+2, 57 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
      # writer.sheets['Transition Quality'].set_row(index + 2, None, top_border)
      writer.sheets['Transition Quality'].freeze_panes(1, 0)
      if file_group_delimiter is not None and file_group_suffix_type is not None:
        type1_col_t = 'F'
        type2_col_t = 'G'
        type3_col_t = 'H'
        qv_start_col = 12
      else:
        type1_col_t = 'D'
        type2_col_t = 'E'
        type3_col_t = 'F'
        qv_start_col = 10
      writer.sheets['Transition Quality'].conditional_format(type1_col_t + '2:' + type1_col_t + str(transition_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
      # writer.sheets['Transition Quality'].conditional_format(type1_col_t + '2:' + type1_col_t + str(transition_df_row_num + 1), type1_color_format)
      writer.sheets['Transition Quality'].conditional_format(type1_col_t + '2:' + type1_col_t + str(transition_df_row_num + 1), {'type': 'cell', 'criteria': 'between', 'minimum': type1_color_format['min_value'], 'maximum': type1_color_format['max_value'], 'format': warn_format})
      writer.sheets['Transition Quality'].conditional_format(type1_col_t + '2:' + type1_col_t + str(transition_df_row_num + 1), {'type': 'cell', 'criteria': 'greater than or equal to', 'value': type1_color_format['max_value'], 'format': good_format})
      writer.sheets['Transition Quality'].conditional_format(type1_col_t + '2:' + type1_col_t + str(transition_df_row_num + 1), {'type': 'cell', 'criteria': 'less than', 'value': type1_color_format['min_value'], 'format': bad_format})
      

      writer.sheets['Transition Quality'].conditional_format(type2_col_t + '2:' + type2_col_t + str(transition_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
      # writer.sheets['Transition Quality'].conditional_format(type2_col_t + '2:' + type2_col_t + str(transition_df_row_num + 1), type2_color_format)
      writer.sheets['Transition Quality'].conditional_format(type2_col_t + '2:' + type2_col_t + str(transition_df_row_num + 1), {'type': 'cell', 'criteria': 'between', 'minimum': type2_color_format['min_value'], 'maximum': type2_color_format['max_value'], 'format': warn_format})
      writer.sheets['Transition Quality'].conditional_format(type2_col_t + '2:' + type2_col_t + str(transition_df_row_num + 1), {'type': 'cell', 'criteria': 'greater than or equal to', 'value': type2_color_format['max_value'], 'format': good_format})
      writer.sheets['Transition Quality'].conditional_format(type2_col_t + '2:' + type2_col_t + str(transition_df_row_num + 1), {'type': 'cell', 'criteria': 'less than', 'value': type2_color_format['min_value'], 'format': bad_format})
      

      writer.sheets['Transition Quality'].conditional_format(type3_col_t + '2:' + type3_col_t + str(transition_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
      #writer.sheets['Transition Quality'].conditional_format(type3_col_t + '2:' + type3_col_t + str(transition_df_row_num + 1), type3_color_format)
      writer.sheets['Transition Quality'].conditional_format(type3_col_t + '2:' + type3_col_t + str(transition_df_row_num + 1), {'type': 'cell', 'criteria': 'between', 'minimum': type3_color_format['min_value'], 'maximum': type3_color_format['max_value'], 'format': warn_format})
      writer.sheets['Transition Quality'].conditional_format(type3_col_t + '2:' + type3_col_t + str(transition_df_row_num + 1), {'type': 'cell', 'criteria': 'greater than or equal to', 'value': type3_color_format['max_value'], 'format': good_format})
      writer.sheets['Transition Quality'].conditional_format(type3_col_t + '2:' + type3_col_t + str(transition_df_row_num + 1), {'type': 'cell', 'criteria': 'less than', 'value': type3_color_format['min_value'], 'format': bad_format})
      

      writer.sheets['Transition Quality'].conditional_format(1, qv_start_col - 4, transition_df_row_num + 1, qv_start_col - 4, peaktop_color_format)
      writer.sheets['Transition Quality'].conditional_format(1, qv_start_col - 3, transition_df_row_num + 1, qv_start_col - 3, peaktop_color_format)
      writer.sheets['Transition Quality'].conditional_format(1, qv_start_col - 2, transition_df_row_num + 1, qv_start_col - 2, peakarea_color_format)
      writer.sheets['Transition Quality'].conditional_format(1, qv_start_col - 1, transition_df_row_num + 1, qv_start_col - 1, peakarea_color_format)
      qv_titles = self.peak_qc.serializeLevel1Title() + self.peak_qc.serializeLevel2Title() + self.peak_qc.serializeLevel3Title()
      for i in range(len(qv_titles)):
        col_idx = qv_start_col + i
        qv_title = qv_titles[i]
        quality_range = self.quality_warn_range_config.get(qv_title)
        writer.sheets['Transition Quality'].conditional_format(1, col_idx, transition_df_row_num + 1, col_idx, {
          'type': '2_color_scale',
          'min_value': quality_range.get('min'),
          'max_value': quality_range.get('max'),
          'min_type': 'num',
          'max_type': 'num',
          'min_color': '#FF0000' if quality_range.get('direction') > 0 else '#00FF00',
          'max_color': '#00FF00' if quality_range.get('direction') > 0 else '#FF0000'
        })

  def output_sample_quality(self, output_path, file_group_delimiter=None, file_group_suffix_type=None, options=None):
    self.summarize_dataset(file_group_delimiter=file_group_delimiter, file_group_suffix_type=file_group_suffix_type)
    sample_df = self._sample_df
    sample_df_row_num = sample_df.shape[0]
    sample_df.rename(columns = {'peptide':'molecule'}, inplace = True)
    print('Output sample qualities: ' + output_path)
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
      sample_df.to_excel(writer, sheet_name='Sample Quality', index=False)
      if 'group' in sample_df:
        grouped_df = sample_df.groupby(['molecule', 'group']).median()
        grouped_df.to_excel(writer, sheet_name='Group Quality', index=True)
      else:
        print('Output Sample and Group Quality file: ' + output_path)
      workbook = writer.book
      workbook_formats = self._get_workbook_formats(workbook, options)
      type1_color_format = workbook_formats['type1_color_format']
      type2_color_format = workbook_formats['type2_color_format']
      type3_color_format = workbook_formats['type3_color_format']
      peaktop_color_format = workbook_formats['peaktop_color_format']
      peakarea_color_format = workbook_formats['peakarea_color_format']
      title_format = workbook_formats['title_format']
      digit_format = workbook_formats['digit_format']
      text_center = workbook_formats['text_center']
      top_border = workbook_formats['top_border']
      left_thin_border = workbook_formats['left_thin_border']
      left_thick_border = workbook_formats['left_thick_border']
      thick_left_top_border = workbook_formats['thick_left_top_border']
      thin_left_top_border = workbook_formats['thin_left_top_border']
      good_format = workbook_formats['good_format']
      warn_format = workbook_formats['warn_format']
      bad_format = workbook_formats['bad_format']
      writer.sheets['Sample Quality'].set_column(1, 20, None, text_center)
      col_idx_shift = 2 if 'group' in sample_df else 0
      for idx, column in enumerate(sample_df):
        writer.sheets['Sample Quality'].write(0, idx, column, title_format)
        if idx <= 2 + col_idx_shift:
          column_len = max(sample_df[column].astype(str).map(len).max(), len(column))
          writer.sheets['Sample Quality'].set_column(idx, idx, column_len + 2)
        else:
          writer.sheets['Sample Quality'].set_column(idx, idx, 10)
      writer.sheets['Sample Quality'].set_column(2 + col_idx_shift, 2 + col_idx_shift, None, left_thick_border)
      writer.sheets['Sample Quality'].set_column(5 + col_idx_shift, 5 + col_idx_shift, None, left_thick_border)
      writer.sheets['Sample Quality'].set_column(7 + col_idx_shift, 7 + col_idx_shift, None, left_thin_border)
      writer.sheets['Sample Quality'].set_column(9 + col_idx_shift, 9 + col_idx_shift, None, left_thick_border)
      writer.sheets['Sample Quality'].set_column(11 + col_idx_shift, 11 + col_idx_shift, None, left_thick_border)
      writer.sheets['Sample Quality'].set_column(13 + col_idx_shift, 13 + col_idx_shift, None, left_thick_border)
      writer.sheets['Sample Quality'].set_column(15 + col_idx_shift, 15 + col_idx_shift, None, left_thick_border)
      temp_id = None
      for index, row in sample_df.iterrows():
        if 'basename' in row:
          id = row['molecule'] + row['basename']
        else:
          id = row['molecule']
        if temp_id is None:
          temp_id = id
        elif temp_id != id:
          writer.sheets['Sample Quality'].set_row(index + 1, None, top_border)
          writer.sheets['Sample Quality'].conditional_format(index + 1, 2 + col_idx_shift, index+1, 2 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
          writer.sheets['Sample Quality'].conditional_format(index + 1, 5 + col_idx_shift, index+1, 5 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
          writer.sheets['Sample Quality'].conditional_format(index + 1, 7 + col_idx_shift, index+1, 7 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
          writer.sheets['Sample Quality'].conditional_format(index + 1, 9 + col_idx_shift, index+1, 9 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
          writer.sheets['Sample Quality'].conditional_format(index + 1, 11 + col_idx_shift, index+1, 11 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
          writer.sheets['Sample Quality'].conditional_format(index + 1, 13 + col_idx_shift, index+1, 13 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
          writer.sheets['Sample Quality'].conditional_format(index + 1, 15 + col_idx_shift, index+1, 15 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
          temp_id = id
      writer.sheets['Sample Quality'].set_row(index + 2, None, top_border)
      writer.sheets['Sample Quality'].conditional_format(index + 2,  2 + col_idx_shift, index+2,  2 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
      writer.sheets['Sample Quality'].conditional_format(index + 2,  5 + col_idx_shift, index+2,  5 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
      writer.sheets['Sample Quality'].conditional_format(index + 2,  7 + col_idx_shift, index+2,  7 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
      writer.sheets['Sample Quality'].conditional_format(index + 2,  9 + col_idx_shift, index+2,  9 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
      writer.sheets['Sample Quality'].conditional_format(index + 2, 11 + col_idx_shift, index+2, 11 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
      writer.sheets['Sample Quality'].conditional_format(index + 2, 13 + col_idx_shift, index+2, 13 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
      writer.sheets['Sample Quality'].conditional_format(index + 2, 15 + col_idx_shift, index+2, 15 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
      # writer.sheets['Sample Quality'].set_row(index + 2, None, top_border)
      writer.sheets['Sample Quality'].freeze_panes(1, 0)

      if 'group' in sample_df:
        # writer.sheets['Group Quality'].set_column(1, 20, None, text_center)
        writer.sheets['Group Quality'].write(0, 0, 'Molecule', title_format)
        writer.sheets['Group Quality'].write(0, 1, 'Group', title_format)
        for idx, column in enumerate(grouped_df):
          writer.sheets['Group Quality'].write(0, idx + 2, column, title_format)
          column_len = max(grouped_df[column].astype(str).map(len).max(), len(column))
          writer.sheets['Group Quality'].set_column(idx+2, idx+2, column_len)
        writer.sheets['Group Quality'].set_column(0, 0, max(list(map(lambda x: len(x[0]), grouped_df.index))))
        temp_id = None
        row_idx = 0
        for index, row in grouped_df.iterrows():
          row_idx += 1
          id = index[0]
          if temp_id is None:
            temp_id = id
          elif temp_id != id:
            writer.sheets['Group Quality'].set_row(row_idx, 14, top_border)
            temp_id = id
        writer.sheets['Group Quality'].set_row(row_idx + 1, 14, top_border)
        writer.sheets['Group Quality'].freeze_panes(1, 0)
      if file_group_delimiter is not None and file_group_suffix_type is not None:
        type1_col = 'E'
        type2_col = 'F'
        type3_col = 'G'
        qv_start_col_sample = 6
      else:
        type1_col = 'C'
        type2_col = 'D'
        type3_col = 'E'
        qv_start_col_sample = 4

      writer.sheets['Sample Quality'].conditional_format(type1_col + '2:' + type1_col + str(sample_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
      #writer.sheets['Sample Quality'].conditional_format(type1_col + '2:' + type1_col + str(sample_df_row_num + 1), type1_color_format)
      writer.sheets['Sample Quality'].conditional_format(type1_col + '2:' + type1_col + str(sample_df_row_num + 1), {'type': 'cell', 'criteria': 'between', 'minimum': type1_color_format['min_value'], 'maximum': type1_color_format['max_value'], 'format': warn_format})
      writer.sheets['Sample Quality'].conditional_format(type1_col + '2:' + type1_col + str(sample_df_row_num + 1), {'type': 'cell', 'criteria': 'greater than or equal to', 'value': type1_color_format['max_value'], 'format': good_format})
      writer.sheets['Sample Quality'].conditional_format(type1_col + '2:' + type1_col + str(sample_df_row_num + 1), {'type': 'cell', 'criteria': 'less than', 'value': type1_color_format['min_value'], 'format': bad_format})
      

      writer.sheets['Sample Quality'].conditional_format(type2_col + '2:' + type2_col + str(sample_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
      #writer.sheets['Sample Quality'].conditional_format(type2_col + '2:' + type2_col + str(sample_df_row_num + 1), type2_color_format)
      writer.sheets['Sample Quality'].conditional_format(type2_col + '2:' + type2_col + str(sample_df_row_num + 1), {'type': 'cell', 'criteria': 'between', 'minimum': type2_color_format['min_value'], 'maximum': type2_color_format['max_value'], 'format': warn_format})
      writer.sheets['Sample Quality'].conditional_format(type2_col + '2:' + type2_col + str(sample_df_row_num + 1), {'type': 'cell', 'criteria': 'greater than or equal to', 'value': type2_color_format['max_value'], 'format': good_format})
      writer.sheets['Sample Quality'].conditional_format(type2_col + '2:' + type2_col + str(sample_df_row_num + 1), {'type': 'cell', 'criteria': 'less than', 'value': type2_color_format['min_value'], 'format': bad_format})
      

      writer.sheets['Sample Quality'].conditional_format(type3_col + '2:' + type3_col + str(sample_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
      #writer.sheets['Sample Quality'].conditional_format(type3_col + '2:' + type3_col + str(sample_df_row_num + 1), type3_color_format)
      writer.sheets['Sample Quality'].conditional_format(type3_col + '2:' + type3_col + str(sample_df_row_num + 1), {'type': 'cell', 'criteria': 'between', 'minimum': type3_color_format['min_value'], 'maximum': type3_color_format['max_value'], 'format': warn_format})
      writer.sheets['Sample Quality'].conditional_format(type3_col + '2:' + type3_col + str(sample_df_row_num + 1), {'type': 'cell', 'criteria': 'greater than or equal to', 'value': type3_color_format['max_value'], 'format': good_format})
      writer.sheets['Sample Quality'].conditional_format(type3_col + '2:' + type3_col + str(sample_df_row_num + 1), {'type': 'cell', 'criteria': 'less than', 'value': type3_color_format['min_value'], 'format': bad_format})
      

      writer.sheets['Sample Quality'].conditional_format(1, qv_start_col_sample + 0, sample_df_row_num + 1, qv_start_col_sample + 0, peaktop_color_format)
      writer.sheets['Sample Quality'].conditional_format(1, qv_start_col_sample + 1, sample_df_row_num + 1, qv_start_col_sample + 1, peaktop_color_format)
      writer.sheets['Sample Quality'].conditional_format(1, qv_start_col_sample + 2, sample_df_row_num + 1, qv_start_col_sample + 2, peakarea_color_format)
      writer.sheets['Sample Quality'].conditional_format(1, qv_start_col_sample + 3, sample_df_row_num + 1, qv_start_col_sample + 3, peakarea_color_format)
      
      if 'group' in sample_df:
        grouped_df_row_num = grouped_df.shape[0]
        writer.sheets['Group Quality'].conditional_format('C2:C' + str(grouped_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
        #writer.sheets['Group Quality'].conditional_format('C2:C' + str(grouped_df_row_num + 1), type1_color_format)
        writer.sheets['Group Quality'].conditional_format('C2:C' + str(grouped_df_row_num + 1), {'type': 'cell', 'criteria': 'greater than or equal to', 'value': type1_color_format['max_value'], 'format': good_format})
        writer.sheets['Group Quality'].conditional_format('C2:C' + str(grouped_df_row_num + 1), {'type': 'cell', 'criteria': 'less than', 'value': type1_color_format['min_value'], 'format': bad_format})
        writer.sheets['Group Quality'].conditional_format('C2:C' + str(grouped_df_row_num + 1), {'type': 'cell', 'criteria': 'between', 'minimum': type1_color_format['min_value'], 'maximum': type1_color_format['max_value'], 'format': warn_format})

        writer.sheets['Group Quality'].conditional_format('D2:D' + str(grouped_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
        #writer.sheets['Group Quality'].conditional_format('D2:D' + str(grouped_df_row_num + 1), type2_color_format)
        writer.sheets['Group Quality'].conditional_format('D2:D' + str(grouped_df_row_num + 1), {'type': 'cell', 'criteria': 'greater than or equal to', 'value': type2_color_format['max_value'], 'format': good_format})
        writer.sheets['Group Quality'].conditional_format('D2:D' + str(grouped_df_row_num + 1), {'type': 'cell', 'criteria': 'less than', 'value': type2_color_format['min_value'], 'format': bad_format})
        writer.sheets['Group Quality'].conditional_format('D2:D' + str(grouped_df_row_num + 1), {'type': 'cell', 'criteria': 'between', 'minimum': type2_color_format['min_value'], 'maximum': type2_color_format['max_value'], 'format': warn_format})

        writer.sheets['Group Quality'].conditional_format('E2:E' + str(grouped_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
        #writer.sheets['Group Quality'].conditional_format('E2:E' + str(grouped_df_row_num + 1), type3_color_format)
        writer.sheets['Group Quality'].conditional_format('E2:E' + str(grouped_df_row_num + 1), {'type': 'cell', 'criteria': 'greater than or equal to', 'value': type3_color_format['max_value'], 'format': good_format})
        writer.sheets['Group Quality'].conditional_format('E2:E' + str(grouped_df_row_num + 1), {'type': 'cell', 'criteria': 'less than', 'value': type3_color_format['min_value'], 'format': bad_format})
        writer.sheets['Group Quality'].conditional_format('E2:E' + str(grouped_df_row_num + 1), {'type': 'cell', 'criteria': 'between', 'minimum': type3_color_format['min_value'], 'maximum': type3_color_format['max_value'], 'format': warn_format})

        writer.sheets['Group Quality'].conditional_format('F2:G' + str(grouped_df_row_num + 1), peaktop_color_format)
        writer.sheets['Group Quality'].conditional_format('H2:I' + str(grouped_df_row_num + 1), peakarea_color_format)

  def _organize_dataset(self):
    store = self.cache_store
    qc_type1_values = list(map(lambda x: x['data'], self.qc_values[0]))
    qc_type2_values = list(map(lambda x: x['data'], self.qc_values[1]))
    qc_type3_values = list(map(lambda x: x['data'], self.qc_values[2]))
    for index, qc_lv1 in enumerate(self.qc_values[0]):
      file = qc_lv1['file']
      peptide = qc_lv1['peptide']
      transition = qc_lv1['transition']
      qc_type1_score = self.qc_scores[0][0][index]
      qc_type1_latent = self.qc_scores[0][1][index]
      qc_type1_value = qc_type1_values[index]
      if peptide not in store:
        store[peptide] = dict()
      if file not in store[peptide]:
        store[peptide][file] = dict()
      if transition not in store[peptide][file]:
        store[peptide][file][transition] = dict()
      store[peptide][file][transition]['type1_score'] = qc_type1_score
      store[peptide][file][transition]['type1_value'] = qc_type1_value
      store[peptide][file][transition]['type1_latent'] = qc_type1_latent
      
    for index, qc_lv3 in enumerate(self.qc_values[2]):
      file = qc_lv3['file']
      peptide = qc_lv3['peptide']
      transition = qc_lv3['transition']
      qc_type3_score = self.qc_scores[2][0][index]
      qc_type3_latent = self.qc_scores[2][1][index]
      qc_type3_value = qc_type3_values[index]
      if peptide not in store:
        store[peptide] = dict()
      if file not in store[peptide]:
        store[peptide][file] = dict()
      if transition not in store[peptide][file]:
        store[peptide][file][transition] = dict()
      store[peptide][file][transition]['type3_score'] = qc_type3_score
      store[peptide][file][transition]['type3_value'] = qc_type3_value
      store[peptide][file][transition]['type3_latent'] = qc_type3_latent
    for index, qc_lv2 in enumerate(self.qc_values[1]):
      file = qc_lv2['file']
      peptide = qc_lv2['peptide']
      qc_type2_score = self.qc_scores[1][0][index]
      qc_type2_latent = self.qc_scores[1][1][index]
      qc_type2_value = qc_type2_values[index]
      if peptide not in store:
          store[peptide] = dict()
      if file not in store[peptide]:
          store[peptide][file] = dict()
      transitions = store[peptide][file].keys()
      for transition in transitions:
        if transition not in store[peptide][file]:
          store[peptide][file][transition] = dict()
        store[peptide][file][transition]['type2_score'] = qc_type2_score
        store[peptide][file][transition]['type2_value'] = qc_type2_value
        store[peptide][file][transition]['type2_latent'] = qc_type2_latent

  def _append_peak_top_and_area(self):
    store = self.cache_store
    filename_list = self.peak_qc.filename_list
    param_list = []
    for peptide in store.keys():
      for file in filename_list:
        param_list.append(dict(qc=self.peak_qc, peptide=peptide, file=file))
    pool = Pool(self.core_num)
    results = pool.map(_get_peak_top_and_area, param_list)
    pool.close()
    pool.join()
    for r in results:
      peptide = r[0]
      file = r[1]
      transitions = r[2]
      if transitions is None:
        continue
      max_peak_intensity = r[3]
      peak_area = r[4]
      for t in transitions:
        store[peptide][file][t]['light_peak_top'] = np.round(np.nan_to_num(np.log10(max_peak_intensity[t + '.light'])), 2)
        store[peptide][file][t]['heavy_peak_top'] = np.round(np.nan_to_num(np.log10(max_peak_intensity[t + '.heavy'])), 2)
        store[peptide][file][t]['light_peak_area'] = np.round(np.nan_to_num(np.log10(peak_area[t + '.light'][0])), 2)
        store[peptide][file][t]['heavy_peak_area'] = np.round(np.nan_to_num(np.log10(peak_area[t + '.heavy'][0])), 2)
  
  def plot_chromatograms(self, output_file, nrow=6, ncol=6, dpi=200, fig_w=30, fig_h=42, threads = 1, mol_separation = True, type1_min=6.7, type1_max=7.7, type2_min = 5.0, type2_max=7.2, type3_min=7.901, type3_max=8.665):
    print("Plotting chromatograms to " + output_file)
    peak_qc = self.peak_qc
    pdf = PdfPages(output_file)
    chrom_list = []
    pep_list = peak_qc.peptide_list
    file_list = peak_qc.filename_list
    page_size = nrow * ncol
    total_page = 1
    #for pep in pep_list:
    #  for file in file_list:
    pb = self.peak_qc.ms_data.peak_boundary
    # for idx, row in pb.iterrows():
    #   pep = row['Peptide Modified Sequence']
    #   file = row['File Name']
    for pep in pep_list:
      for file in file_list:
        chrom_data = peak_qc.selectChromData(file, pep)
        if chrom_data is not None:
          chrom_list.append(chrom_data)
        else:
          ms_data = self.peak_qc.ms_data
          raw_chrom = ms_data.chrom[(ms_data.chrom['FileName'] == file) & (ms_data.chrom['PeptideModifiedSequence'] == pep)]
          if len(raw_chrom) > 0:
            rt, inst = ms_data.getChromRTInst(file, pep)
            chrom_data = ms_data.make_chrom_without_boundary(file, pep, rt, inst)
            chrom_list.append(chrom_data)
      if mol_separation:
        current_length = len(chrom_list)
        remainder = len(chrom_list) % page_size
        if remainder > 0:
          chrom_list = chrom_list + ([None] * (page_size - remainder))
            
    total_page = math.floor(len(chrom_list) / page_size)
    if (len(chrom_list) % page_size) > 0:
        total_page += 1
    page_input_options = []
    # print('Total chromatogram plots: ' + str(len(chrom_list)))
    # print('#Chromatograms per page: ' + str(page_size))
    print('Total page: ' + str(total_page))
    for page_idx in range(total_page):
        page_start = page_idx * page_size
        page_end = page_start + page_size
        page_items = chrom_list[page_start:page_end]
        score_tables = dict()
        for chrom_data in page_items:
          if chrom_data is None:
            continue
          fn = chrom_data['fileName']
          ps = chrom_data['peptideModifiedSequence']
          if ps not in score_tables:
            score_tables[ps] = dict()
          if ps in self.cache_store and fn in self.cache_store[ps]:
            score_tables[ps][fn] = pd.DataFrame(self.cache_store[ps][fn]).T
          else:
            score_tables[ps][fn] = None
        page_input_options.append({'page_items': page_items, 'nrow': nrow, 'ncol': ncol, 'score_tables': score_tables,
                                    'dpi': dpi, 'fig_w': fig_w, 'fig_h': fig_h, 'page_num': page_idx + 1, 'total_page': total_page,
                                    'mol_separation': mol_separation, 'type1_min': type1_min, 'type1_max': type1_max, 'switchLightHeavy': self.switchLightHeavy,
                                    'type2_min':type2_min, 'type2_max': type2_max, 'type3_min': type3_min, 'type3_max': type3_max
                                  })
    print('Generating figures ...')
    page_num = 1
    if threads >= 2:
      pool = Pool(threads)
      figs = pool.map(_chrom_plot_page, page_input_options)
      pool.close()
      pool.join()
      print('Writing figures into the pdf: ' + str(output_file))
      for fig in figs:
        pdf.savefig(fig, bbox_inches='tight')
        print('Page ' + str(page_num) + ' of ' + str(total_page) + ' was written.')
        page_num += 1
        plt.clf()
        plt.close(fig)
    else:
      page_num = 1
      print('Writing figures into the pdf: ' + str(output_file))
      for each_page_input in page_input_options:
        fig = _chrom_plot_page(each_page_input)
        pdf.savefig(fig, bbox_inches='tight')
        plt.clf()
        plt.close(fig)
        print('Page ' + str(page_num) + ' of ' + str(total_page) + ' was written.')
        page_num += 1
    pdf.close()

