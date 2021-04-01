# SwissRt
The scripts and data provided in this project are related to the manuscript: C. K. Sruthi, M. R. Biswal, H. Joshi, B. Saraswat, and M. K. Prakash, How Polices on Restaurants, Bars, Nightclubs, Masks, Schools, and Travel Influenced Swiss COVID-19 Reproduction Ratios, MedRxiv (2020)

covid_19_data_switzerland.xslx: The raw data for the daily new infections was obtained from the website: https://www.corona-data.ch/

TransmissionRateEstiamtes: The processed data from the weekly averaged infection rates for the 26 Swiss Cantons is given in this folder. To obtain the Reproduciton Ratios, these transmission ratios should be divided by the recovery rate gamma=1/14.

RawPolicyData_Sources: The policy data given as 3 separate zip files was collected mainly from the news articles published in english language in thelocal.ch, swissinfo.ch, swisskarte.ch, some of which are in this folder.

SwissCanton_NPI_Policies.csv: The policy data for the 26 different cantons between 9 March to 13 September

all_data_combined.csv: The combined data including policies, and weekly averaged rates

randsearchoptparams.py: Script for optimizing the hyper-parameters to be used in the XGBoost AI model

shapanalysis.py: Script for performing the SHAP analysis for identifying the individual policy contributions to the Transmission rates (or reproduction rates, if multiplied by 14, as mentioned above)

Rt_contribution_CSV: Contains all the processed data  
- combine_allVars: Weekly policy data for all the 26 cantons and their corresponding Rt contribution  
- combine_delta_minmax: Change in the Rt contribution with respect to the 20th week
