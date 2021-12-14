# Optical Gating: Retrospective Analysis By Log Scraping

The enclosed scripts function to extract the necessary information from optical-gating log files to perform a retrospective analysis of the quality of the synchonisation.
This will only function for logs generated in optical-gating versions installed/pulled after the 14th of December 2021, when the dependent log statements were introduced. 

### Dependencies

The only package dependencies are 'sys', 'numpy', 'matplotlib', 'time', and 'json'. Correct installation of base optical-gating will provide all dependencies. 

### Running

The scripts are collated in 'scrape_and_plot.py', which requires a log-file-path and log-keys-path.

For example, using relative paths, the script can be run using command:

`python scrape_and_plot.py 'example_log_folder/example_1.log' 'log_keys.json'`