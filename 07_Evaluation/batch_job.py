import os

# examples for macro script calls with different datasets and variables

# SER/SNR sweep
os.system("python macro_script.py -SER -10 -5 0 5 10 -SNR 0 10 20 -d TUB_synth_test8_TIMITrealDT_minS_16 -MOS -m Kalman")
os.system("python macro_script.py -SER -10 -5 0 5 10 -SNR 0 10 20 -d TUB_synth_test8_TIMITrealDT_minS_16 -MOS -m NLMS")

# delay sweep; audio not saved
os.system("python macro_script.py --noaudio -delay 0 -d TUB_synth_test8_TIMITrealDTdc_16 -MOS -m Kalman")
os.system("python macro_script.py --noaudio -delay 200 -d TUB_synth_test8_TIMITrealDTdc_16 -MOS -m Kalman")
os.system("python macro_script.py --noaudio -delay 400 -d TUB_synth_test8_TIMITrealDTdc_16 -MOS -m Kalman")
os.system("python macro_script.py --noaudio -delay 800 -d TUB_synth_test8_TIMITrealDTdc_16 -MOS -m Kalman")

# added AECMOS metric computation
os.system("python macro_script.py -d TUB_synth_test8_TIMITrealDT_minS_16 -MOS -m Kalman")
os.system("python macro_script.py -d TUB_synth_test8_TIMITrealDT_minS_16 -MOS -m NLMS")

# cold start: convergence sections are removed
os.system("python macro_script.py -CS -d TUB_synth_test8_TIMITrealDTdyn_16 -MOS -m Kalman")
os.system("python macro_script.py -CS -d TUB_synth_test8_TIMITrealFEdyn_16 -MOS -m Kalman")