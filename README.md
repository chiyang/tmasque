# tmasque: Targeted Mass Spectrometry Quality Encoder

For processing datastes of targeted mass spectrometry, this tmasque package helps to evaluate peak quality of chromatograms objectively after running automatic peak-picking software. We conducted an unsupervised learning approach, the beta-total correlation variational autoencoder (beta-TCVAE), to learn peak quality from our collected TargetedMS data cohort (including both MRM and PRM experiments). The data cohort contains 1,703,827 peak groups from 75 studies and was produced from 22 different models of mass spectrometers, manufactured by the four leading brands: AB Sciex, Agilent Technologies, Thermo Fisher Scientific, and Waters Corporation. Such versatile datasets can contribute to a reliable quality encoder to provide objective scoring of chromatographic peaks.

This Python package carries a command line tool and provides a programming interface to calculate TMSQE scores and generate friendly reports.

## Installation
The package is published as a python package. You need to have a Python with version >=3.8 installed and can then use `pip` to install tmasque.

```
pip install tmasque

```

Then, you can use `tmasque -h` or `tmasque --help` to see the arguments.
You can also use `tmasque example` to have a quick example (see below).

## Required data format

We currently support the TSV (tab-separated values) format file for chromatograms and the CSV (comma-separated values) format file for peak boundaries. 
**These two files can be directly exported via Skyline.**

The required nine column headers for chromatograms are listed as follows.

| FileName |  PeptideModifiedSequence  |  PrecursorCharge | ProductMz | FragmentIon | ProductCharge | IsotopeLabelType | Times | Intensities|
|----------|---------------------------|------------------|-----------|-------------|---------------|------------------|-------|------------|

For peak boundary CSV file, the following five column headers are required.

| File Name |  Peptide Modified Sequence  |  Min Start Time | Max End Time | Precursor Charge |
|-----------|-----------------------------|-----------------|--------------|------------------|

You may notice that the spaces between words in the above column headers. 
This is because that these column headers followed the Skyline output formats for chromatograms and peak boundaries. 

The `PeptideModifiedSequence` and `Peptide Modified Sequence` columns are used for display and for identifications of analyte targets. For metabolites, you may ignore the header and using metabolite name for this column is fine.



We followed the Skyline exported file formats for chromatograms and peak boundaries.

## Command line tool
Once you have installed tmasque, you can use `tmasque` as the command line tool.
In your terminal, type `tmasque --help` to see the detailed descriptions for arguments.
The full arguments are also listed in the next section.

### 1. Quick example

Use the following command to run the quick example.

```Shell
tmasque example
```

We have prepared a small example in this package. After running the `tmasque example` command, a folder will be created in your current working directory.
This folder will also contain input files: `chromatogram.tsv` and `peak_boundary.csv`. You may also take a look the column headers and the formats.
This folder are `sample_quality.xlsx` and `chromatogram_plots.pdf`. The former is the summarized peak quality scores for each sample, and the latter is the chromatogram plots.
Also, this folder has a `transition_quality` folder in it. You can see one excel file for each target. The excel file displays the transition quality scores with all the underlying quality features.


### 2. The basic usage:

```Shell
tmasque run <chromatogram_path> <peak_boundary_path> <output_sample_quality_excel_path> <output_transition_quality_folder_path>
```

In the above command, the former two arguments `chromatogram_path` and `peak_boundary_path` are the two required input files of the TSV and CSV formats, while the latter two are the output paths. `output_sample_quality_excel_path` expects an Excel file path and `output_transition_quality_folder_path` expects a folder path. If the folder path does not exist, the folder will be created during tmasque execution. In this folder, detailed quality feature values of each target would be exported in an Excel file. 



### 3. Quality scores with additional chromatogram plots
We offer an option to plot chromatograms for visual examinations and easy identifications of problematic cases.
Use the following command arguments to perform quality evaluation and generate chromatogram plots.

```Shell
tmasque run <chromatogram_path> <peak_boundary_path> <output_sample_quality_excel_path> <output_transition_quality_folder_path> --output_chromatogram_pdf<output_chromatogram_pdf_path>
```

By adding the optional argument `--output_chromatogram_pdf` with the value of the expected output PDF path, a PDF file will be generated to display all chromatogram plots with color indications of good, acceptable, and poor peak quality.

### 4. Sample grouping for batches or response curve experiments
We can group a couple of samples and summarize the peak quality scores with medians to examine batche quality or the peak quality at different concentrations for response curve experiments.
To deal with sample grouping, we can use arguments of `file_group_delimiter` and `file_group_suffix_type` to specify the rules for grouping.
For example, if the sample names are listed as follows, we can use `file_group_delimiter=-` and `file_group_suffix_type='d'` to extract the suffix of the filename as the group name.

| Sample name| Group|
|:-----------|-----:|
|20210727_DSS Response curve_121plex_150um_1-001.wiff|1|
|20210727_DSS Response curve_121plex_150um_2-001.wiff|1|
|20210727_DSS Response curve_121plex_150um_1-002.wiff|2|
|20210727_DSS Response curve_121plex_150um_2-002.wiff|2|

Once the `file_group_delimiter` and `file_group_suffix_type` are set, quality scores will be further summarized from sample quality.
An additional sheet `Group Quality` will be created in the `output_sample_quality_excel_path` file.

## Full command argument list for `tmasque run`

| Argument      |   | Description | Value Type |Default Values<tr><td colspan="5">**INPUT**</td></tr>
|:--------|--|:------------|------------|----------:|
|chromatogram_tsv| |The chromatogram TSV file path| File path|no default|
|peak_boundary_csv| |The Peak Boundary CSV file path|File path|no default<tr><td colspan="5">**OUTPUT**</td></tr>
|output_quality_xlsx||The output peak quality Excel file path|File path|no default|
|output_transition_folder| |The output transition quality folder path|Folder path|no default<tr><td colspan="5">**OPTIONAL**</td></tr>
|--help| -h |Show the detailed argument list|(no value)|unset|
|--version| -v |Display the package version|(no value)|unset|
|--thread_num|-t|The parallel thread number to calculate quality feature values for all peak groups| integer |4|
|--internal_standard_type|-s|Set the internal standards to heavy or light ions.|{`heavy`, `light`}|heavy<tr><td colspan="5">**GROUPING**</td></tr>
|--file_group_delimiter|-gs|The delimiter of filename to define file groups| {`None`, `-`, `_`} | None|
|--file_group_suffix_type|-gt|The suffix character type to define file groups (d for digit number; w for words)|{`d`, `w`}|None<tr><td colspan="5">**SCORING**</td></tr>
|--type1_acceptable_threshold|-t1a|The threshold of acceptable quality of Type 1 quality score. Scores greater than or equal to threshold value are considered as poor quality.|float| 6.7|
|--type1_good_threshold|-t1b|The threshold of good quality of Type 1 quality score. Scores greater or equal to this threshold are considered as good quality.|float| 7.7|
|--type2_acceptable_threshold|-t2a|The threshold of acceptable quality of Type 2 quality score. Scores greater than or equal to threshold value are considered as poor quality.|float| 5.0|
|--type2_good_threshold|-t2b|The threshold of good quality of Type 2 quality score. Scores greater or equal to this threshold are considered as good quality.|float| 7.2 |
|--type3_acceptable_threshold|-t3a|The threshold of acceptable quality of Type 3 quality score. Scores greater than or equal to threshold value are considered as poor quality.|float| 7.901|
|--type3_good_threshold|-t3b|The threshold of good quality of Type 3 quality score. Scores greater or equal to this threshold are considered as good quality.|float|8.665<tr><td colspan="5">**CHROMATOGRAM PLOTS**</td></tr>
|--output_chromatogram_pdf|-plot|The output chromatogram pdf path to draw chromatogram plots. If set, all chromatograms will be plotted in a PDF file.|File path|None|
|--output_mixed_mol|-mix| If set, chromatogram data for each target will be mixed in one pdf page.|(no value)|unset|
|--output_chromatogram_nrow|-nrow|The number of chromatograms per row in one pdf page|integer|6|
|--output_chromatogram_ncol|-ncol|The number of chromatograms per columne in one pdf page |integer|6|
|--output_chromatogram_fig_w|-figw|The figure width in inches|integer|30|
|--output_chromatogram_fig_h|-figh|The figure height in inches|integer|42|
|--output_chromatogram_dpi|-dpi|The dpi of chromatogram plots|integer|200|
|--output_chromatogram_threads|-pthread|Using more threads will consume more memory|integer|1|



## Use tmasque as a module
The tmasque package can also be used as a module to integrate quality scoring functions in Python scripts.

```python
from tmasque import TargetedMSQualityEncoder as TMSQE
chromatogram_tsv = '/path/to/chromatogram.tsv'
peak_boundary_csv = '/path/to/peak_boundary.csv'

tmsqe = TMSQE(chromatogram_tsv, peak_boundary_csv)
tmsqe.run_encoder()

# Get all quality features for all transitions in transition_df and summarized sample quality in sample_quality_df
transition_df, sample_quality_df = tmsqe.summarize_dataset()


# Output quality results in Excel files
output_transition_folder = '/path/to/a/folder'
output_summarized_sample_quality_xlsx = '/path/to/output_sample_quality.xlsx'
tmsqe.output_transition_quality(output_transition_folder)  # Output quality scores for each transition with all the underlying quality features. One excel per target.
tmsqe.output_sample_quality(output_summarized_sample_quality_xlsx) # Output summarized quality scores for each sample.

# Generate all chromatogram plots in a PDF file.
output_chromatogram_pdf = '/path/to/output_chromatogram_plots.pdf'
tmsqe.plot_chromatograms(output_chromatogram_pdf)
```
### For sample grouping

```python
# for sample names like xxxx-01.raw or xxxx-02.raw, the suffix 01 and 02 will be parsed as numbers (1 and 2) and group quality will be summarized in a different sheet in output_summarized_sample_quality_xlsx.
transition_df, sample_quality_df = tmsqe.summarize_dataset(file_group_delimiter='-', file_group_suffix_type='d')
tmsqe.output_transition(output_transition_folder, file_group_delimiter='-', file_group_suffix_type='d')
tmsqe.output_quality(output_summarized_sample_quality_xlsx, file_group_delimiter='-', file_group_suffix_type='d')

```















