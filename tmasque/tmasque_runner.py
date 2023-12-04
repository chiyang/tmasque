#!/usr/bin/env python
import argparse
import tmasque
import os
from tmasque import TargetedMSQualityEncoder
import shutil

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--version', '-v', help='show the version of the tmasque package', action='version', version=tmasque.__version__)
  subparsers = parser.add_subparsers(description='available subcommands')
  parser_main = subparsers.add_parser('run', help='Run quality evaluations for MRM chromatographic peaks.')
  parser_main.add_argument("chromatogram_tsv", type=str, help="The chromatogram TSV file path")
  parser_main.add_argument("peak_boundary_csv", type=str, help="The Peak Boundary CSV file path")
  parser_main.add_argument("output_quality_xlsx", type=str, help="The output peak quality Excel file path")
  parser_main.add_argument("output_transition_folder", type=str, help="The output transition quality folder path")
  parser_main.add_argument('--thread_num', '-t', type=int, default=4, help="The parallel thread number to calculate quality feature values for all peak groups (default: 4)")
  parser_main.add_argument('--internal_standard_type', '-s',type=str, default='heavy', choices=['heavy', 'light'] , help="Set the internal standards to heavy or light ions. (default: heavy)")
  parser_main.add_argument('--file_group_delimiter', '-gs', type=str, default=None, choices=[None, '-', '_'], help="The delimiter of filename to define file groups (default: None)")
  parser_main.add_argument('--file_group_suffix_type', '-gt', type=str, default=None, choices=[None, 'd', 'w'], help="The suffix character type to define file groups (d for number and w for word, default: None)")
  parser_main.add_argument('--type1_acceptable_threshold', '-t1a', type=float, default=6.7, help="The threshold of acceptable quality of Type 1 quality score. Scores greater than or equal to threshold value are considered as poor quality. (default: 6.7)")
  parser_main.add_argument('--type1_good_threshold', '-t1b', type=float, default=7.7, help="The threshold of good quality of Type 1 quality score. Scores greater or equal to this threshold are considered as good quality. (default: 7.7)")
  parser_main.add_argument('--type2_acceptable_threshold', '-t2a', type=float, default=5.0, help="The threshold of acceptable quality of Type 2 quality score. Scores greater than or equal to threshold value are considered as poor quality. (default: 5.0)")
  parser_main.add_argument('--type2_good_threshold', '-t2b', type=float, default=7.2, help="The threshold of good quality of Type 2 quality score. Scores greater or equal to this threshold are considered as good quality. (default: 7.2)")
  parser_main.add_argument('--type3_acceptable_threshold', '-t3a', type=float, default=7.901, help="The threshold of acceptable quality of Type 3 quality score. Scores greater than or equal to threshold value are considered as poor quality. (default: 7.901)")
  parser_main.add_argument('--type3_good_threshold', '-t3b', type=float, default=8.665, help="The threshold of good quality of Type 3 quality score. Scores greater or equal to this threshold are considered as good quality. (default: 8.665)")
  parser_main.add_argument('--output_chromatogram_pdf', '-plot', type=str, default=None, help="The output chromatogram pdf path to draw chromatogram plots. If set, all chromatograms will be plotted in a PDF file. (default: None)")
  parser_main.add_argument('--output_chromatogram_threads', '-pthread', type=int, default=1, help="Using more threads will consume more memory. (default: 1)")
  parser_main.add_argument('--output_chromatogram_nrow', '-nrow', type=int, default=6, help="The number of chromatograms per row in one pdf page. Only works when --output_chromatogram_pdf is set. (default: 6)")
  parser_main.add_argument('--output_chromatogram_ncol', '-ncol', type=int, default=6, help="The number of chromatograms per columne in one pdf page. Only works when --output_chromatogram_pdf is set. (default: 6)")
  parser_main.add_argument('--output_chromatogram_dpi', '-dpi', type=int, default=200, help="The dpi of chromatogram plots. Only works when --output_chromatogram_pdf is set. (default: 200)")
  parser_main.add_argument('--output_chromatogram_fig_w', '-figw', type=int, default=30, help="The figure width in inches. Only works when --output_chromatogram_pdf is set. (default: 30)")
  parser_main.add_argument('--output_chromatogram_fig_h', '-figh', type=int, default=42, help="The figure height in inches. Only works when --output_chromatogram_pdf is set. (default: 42)")
  parser_main.add_argument('--output_mixed_mol', '-mix', action='store_true', help="If set, chromatogram data for each molecule will be mixed in one pdf page. (default: unset)")
  parser_main.add_argument('--encoder_type1_path', type=str, default=None, help='Custom Type1 encoder pickle path')
  parser_main.add_argument('--encoder_type2_path', type=str, default=None, help='Custom Type2 encoder pickle path')
  parser_main.add_argument('--encoder_type3_path', type=str, default=None, help='Custom Type3 encoder pickle path')
  parser_main.add_argument('--encoder_dim', type=int, default=2, help='Custom dimension for latent space')

  
  parser_main.set_defaults(example=False)
  parser_example = subparsers.add_parser('example', help='A quick example will be run. An output folder will be created in the current working directory. The folder contains input files (chromatogram.tsv and PeakBoundar.csv) and output data.')
  parser_example.set_defaults(
    chromatogram_tsv=os.path.join(os.getcwd(), 'tmasque_output', 'chromatogram.tsv'), 
    peak_boundary_csv=os.path.join(os.getcwd(), 'tmasque_output', 'peak_boundary.csv'),
    output_quality_xlsx=os.path.join(os.getcwd(), 'tmasque_output', 'sample_quality.xlsx'),
    output_transition_folder=os.path.join(os.getcwd(), 'tmasque_output', 'transition_quality'),
    output_chromatogram_pdf=os.path.join(os.getcwd(), 'tmasque_output', 'chromatogram_plots.pdf'),
    example=True,
    internal_standard_type='heavy',
    thread_num=4,
  )
  args = parser.parse_args()
  print("Input Parameters:")
  for arg in vars(args):
    print(arg, '=',getattr(args, arg))
  print('#########################################################')

  #switch_light_heavy = True if args.switch_light_heavy==1  else False
  switch_light_heavy = True if args.internal_standard_type == 'light' else False
  if args.example:
    # Examle Data from https://panoramaweb.org/RAS_mAb_resource.url
    os.makedirs(os.path.join(os.getcwd(), 'tmasque_output'), exist_ok=True)
    shutil.copyfile(os.path.join(os.path.split(__file__)[0], 'example', 'CPTC-EGFR-5_YSSD_2.tsv'), args.chromatogram_tsv)
    shutil.copyfile(os.path.join(os.path.split(__file__)[0], 'example', 'PeakBoundaries.csv'), args.peak_boundary_csv)
    tmsqe = TargetedMSQualityEncoder(args.chromatogram_tsv, args.peak_boundary_csv, core_num=args.thread_num, switchLightHeavy=switch_light_heavy)
    tmsqe.run_encoder()
    tmsqe.output_transition_quality(args.output_transition_folder)
    tmsqe.output_sample_quality(args.output_quality_xlsx)
    tmsqe.plot_chromatograms(args.output_chromatogram_pdf)
  else:
    tmsqe = TargetedMSQualityEncoder(args.chromatogram_tsv, args.peak_boundary_csv, core_num=args.thread_num, switchLightHeavy=switch_light_heavy)
    tmsqe.run_encoder(encoder_type1_path = args.encoder_type1_path, encoder_type2_path=args.encoder_type2_path, encoder_type3_path=args.encoder_type3_path, encoder_dim=[args.encoder_dim, args.encoder_dim, args.encoder_dim])
    tmsqe.output_transition_quality(
      args.output_transition_folder,
      file_group_delimiter=args.file_group_delimiter,
      file_group_suffix_type=args.file_group_suffix_type,
      options=dict(
        type1_min=args.type1_acceptable_threshold,
        type1_max=args.type1_good_threshold,
        type2_min=args.type2_acceptable_threshold,
        type2_max=args.type2_good_threshold,
        type3_min=args.type3_acceptable_threshold,
        type3_max=args.type3_good_threshold
      ))
    tmsqe.output_sample_quality(
      args.output_quality_xlsx,
      file_group_delimiter=args.file_group_delimiter,
      file_group_suffix_type=args.file_group_suffix_type,
      options=dict(
        type1_min=args.type1_acceptable_threshold,
        type1_max=args.type1_good_threshold,
        type2_min=args.type2_acceptable_threshold,
        type2_max=args.type2_good_threshold,
        type3_min=args.type3_acceptable_threshold,
        type3_max=args.type3_good_threshold
      ))
    if args.output_chromatogram_pdf is not None:
      tmsqe.plot_chromatograms(args.output_chromatogram_pdf, nrow=args.output_chromatogram_nrow, ncol=args.output_chromatogram_ncol, dpi=args.output_chromatogram_dpi, fig_w=args.output_chromatogram_fig_w, fig_h=args.output_chromatogram_fig_h, threads=args.output_chromatogram_threads, mol_separation=not args.output_mixed_mol,
                            type1_min=args.type1_acceptable_threshold, type1_max=args.type1_good_threshold, type2_min=args.type2_acceptable_threshold, type2_max=args.type2_good_threshold, type3_min=args.type3_acceptable_threshold, type3_max=args.type3_good_threshold)
  print('Finished')
if __name__ == '__main__':
  main()

  