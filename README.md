# Acoustic Echo Control Evaluation Toolbox



## 1. Introduction

Acoustic echo cancellation (AEC) and suppression (AES) are widely researched topics. However, only few papers about hybrid or deep acoustic echo control provide a solid comparative analysis of their methods as it was common with classical signal processing approaches. There can be distinct differences in the behaviour of an AEC/AES model which cannot be fully represented by a single metric or test condition, especially when comparing classical signal processing and machine-learned approaches. These characteristics include convergence behaviour, reliability under varying speech levels or far-end (FE) signal types, as well as robustness to adverse conditions such as harsh nonlinearities, room impulse response switches or continuous changes, or delayed echo. 
We provide a toolbox that allows evaluation on an extended set of test conditions and metrics, mainly focussed around the application on 16 kHz signals.

![AEC/AES system](Images/AEC_AES.PNG)

We refer to system signals as shown above. The FE reference signal x(n) is played by the loudspeaker and picked up by the microphone as echo d(n) alongside the near-end (NE) speech signal s(n) and background noise n(n). The microphone signal y(n) is processed by the AEC/AES system, resulting in the enhanced signal e(n).

If you use our toolbox for your research, please cite our work:

Application of the toolbox for AEC/AES analysis (will be updated once finally published):
```BibTex
@Article{Seidel2024,
  author      = {Ernst Seidel and Tim Fingscheidt},
  title       = {{Convergence and Performance Analysis of Classical, Hybrid, and Deep Acoustic Echo Control}},
  journal	= {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year		= {2024},
  volume	= {},
  number	= {},
  pages		= {1-15},
  month		= may,
}
```

Black-box metrics:
```BibTex
@InProceedings{Fingscheidt2007,
  author	= {T. Fing\-scheidt and S. Suhadi},
  booktitle	= {Proc. of Interspeech},
  title		= {{Quality Assessment of Speech Enhancement Systems by Separation of Enhanced Speech, Noise, and Echo}},
  year		= {2007},
  address	= {Antwerp, Belgium},
  month		= aug,
  pages		= {818--821},
  keywords	= {Signal Processing, Measure},
}

@InProceedings{Fingscheidt2008,
  author	= {Tim Fingscheidt and Suhadi Suhadi and K. Steinert},
  title		= {{Towards Objective Quality Assessment of Speech Enhancement Systems in a Black Box Approach}},
  booktitle	= {Proc. of ICASSP},
  year		= {2008},
  pages		= {273--276},
  address	= {Las Vegas, NV, USA},
  month		= apr,
}
```

Braunschweig Dynamic Impulse Response Database:
```BibTex
@InProceedings{Jung2013,
  author      = {Marc-André Jung and Lucca Richter and Tim Fingscheidt},
  title       = {{Towards Reproducible Evaluation of Automotive Hands-Free Systems in Dynamic Conditions}},
  booktitle   = {Proc. of ICASSP},
  year        = {2013},
  pages       = {8144--8148},
  address     = {Vancouver, Canada},
  month       = may,
}
```

### 1.1 Roadmap

We plan to improve our initial published code to be more accessible and intuitive to use by adding more helper scripts and ultimately transitioning to a fully python-based version of the Toolbox.

- [x] GitHub Release
- [ ] Automated script for download and preprocessing of available datasets (where licenses allow)
- [ ] Expand out-of-the box dataset support: generic file discovery for speaker data
- [ ] PyTorch Toolbox: Modular, all-in-one version of the current code...
- [ ] ... including model training 

### 1.2 Prerequisites / Compatibility

Operating system:
All scripts were previously run on Windows 10 machines, and the evaluation scripts was also tested on CentOS.

Matlab:
The data generation in this toolbox was conducted on Matlab 2021b.

Python Requisites:

Install the following python packages (using conda) as shown below:

```bash
conda create -n <your_env> python=3.12
conda activate <your_env>

# torch with GPU support
pip3 install torch --index-url https://download.pytorch.org/whl/cu118 
# torch on CPU
pip3 install torch

conda install numpy scipy
pip3 install soundfile librosa webrtcvad onnxruntime

# custom P.862 Corrigendum 1 python module
pip install git+https://pipuser:9suQnFZm9ENhsFyBFgvn@git.rz.tu-bs.de/ifn/svml/pesq.git
```

## 2. Metrics (and Metric Setup Instructions)

As part of this toolbox, we provide various metrics, usable both as standalone python functions (on preprocessed files) or using the macro script for datasets. Some of the metrics require the download of third party source code as detailed below in order to function. All metrics can be accessed individually, or via the macro script (detailed below).

Note: All metrics can be verified by running the metric_unittest.py file in the Evaluation script folder.

### 2.1 Individual Metrics

#### Perceptual Evaluation of Speech Quality (PESQ)

The perceptual evalution of quality (PESQ) metric is a widely used metric for the assessment of speech quality, standardized as [ITU-T recommendation P.862](https://www.itu.int/rec/T-REC-P.862.2), and regularly used in evaluation of echo control mechanism. Please note that with the presence of unsuppressed background noise, results are expected to yield low scores.

A custom PESQ python module (including Corrigendum 1) can be installed via:

```bash
pip install git+https://pipuser:9suQnFZm9ENhsFyBFgvn@git.rz.tu-bs.de/ifn/svml/pesq.git
```

Please note: If you install the python pesq module (pip install pesq), you need to uninstall it first. The modules are not identical, as ours includes Corrigendum 1.

Once installed correctly, you can call PESQ through our macro script (details below) or by calling:

```python
from pesq import pesq

# signals: enhanced signal [e]; clean NE speech [s]
result = pesq(sampling_rate, s, e)
```

#### Echo Return Loss Enhancement (ERLE)

The echo return loss enhancement (ERLE) metric is commonly used to evaluate the suppression of far-end echo. Our implementation follows [[Vary, 2006]](https://onlinelibrary.wiley.com/doi/book/10.1002/0470031743). It can be accessed via 

```python
from evalution_metrics import compute_ERLE

# signals: enhanced signal [e]; microphone signal [y]; echo component [d]
mean_erle, erle_over_time = compute_ERLE(y, e, d, s_f=0.99)
```

Note that this function provides you with both the mean ERLE score as well as a sample-wise ERLE over time, which is useful for analysis of (re)converegence behaviour.

#### AECMOS Metrics

The AECMOS metric [published by Purin et. al.](https://arxiv.org/abs/2110.03010) is a machine-learned evaluation metric for both near-end speech quality as well as echo control effectiveness. A great advantage of these metrics is their non-intrusiveness, allowing their evaluation on real-world test data. We provide a slightly altered version fitted for our code. The original source code is available at:

[AEC Challenge GitHub](https://github.com/microsoft/AEC-Challenge)

The AECMOS metrics can be calculated via:

```python
import os
from aecmos_local import AECMOSEstimator

# signals: enhanced signal [e]; microphone signal [y]; reference signal [x]
# talk_type in ['nst', 'fst', 'dt']; refering to single-talk far-end, single-talk near-end, and double-talk
aecmos      = AECMOSEstimator(os.getcwd() +'/models/Run_1663915512_Stage_0.onnx')
echo, other = aecmos.run(talk_type='dt', lpb_sig=x,mic_sig=y,enh_sig=e)
```

Note that this function evaluates on the entire sequence and does not contain the trimming of test files present in the original source code. 

#### Log-Spectral Distance (LSD)

The log-spectral distance (LSD) metric (implemented after [[Katsir, 2011]](https://ieeexplore.ieee.org/document/7074052)), much like PESQ, reports overall quality (both NE speech preservation and echo suppression effectiveness affect the score), but is a straightforward distance metric. As such, it is less prone to performance differences getting masked by noise, but in return is less descriptive on the perceptual impact of residual echo and NE speech degradation. It can be accessed via:

```python
from evaluation_metrics import compute_LSD

# signals: enhanced signal [e]; clean NE speech [s]
LSD_scores, LSD_mean = compute_LSD(s, e)
```

### 2.2 Black-Box Metric Variants

PESQ, ERLE, and LSD are also available as black-box variants. The use of the black-box algorithm allows the disentanglement components in the enhanced signal, which can be used for a more precise evaluation of certain performance aspects (e.g., measuring NE PESQ without the disturbing influence of background noise). A visual representation of the black-box algorithm [by the original authors](https://www.isca-archive.org/interspeech_2007/fingscheidt07_interspeech.pdf) is given below:

![Black-box](Images/black_box.PNG)

As can be seen in the figure, the black-box algorithm computes a mask from the microphone signal (based on the enhanced signal,called s_hat(n)) and applies the mask to the individual signal components.

The black-box components can be computed and applied via:

```python
from audio_processing import get_BB_components
from evaluation_metrics import *
from pesq import pesq

params = {'window': window, 'K_fft': 512, 'K_mask': 257, 'frame_length':512, 'frame_shift': 64}


# signals: enhanced signal [e]; microphone signal [y]
# components: NE speech [s], echo [d], noise [n]
[s_tilde, d_tilde, n_tilde] = get_BB_components(e, y, components=[s, d, n], params=params)

result                      = pesq(sampling_rate, s, s_tilde)
mean_erle, erle_over_time   = compute_ERLE(y, s_tilde, d, d_tilde, s_f=0.99)
LSD_score                   = compute_LSD(s, s_tilde)
```

In order for the black-box algorithm to work properly, it is crucial that the indivual components are aligned with their counterpart inside the mixture (this is by default the case for data generated from our macro script). 
Note that compute_erle takes an additional argument in black-box configuration.
While the black-box parameters are adjustable, it is recommended to leave them as is for optimal component disentanglement.

### 2.3 Evaluation Macro Script

The evaluation of all metrics over a large number of files can be automated by the provided macro script score_AEC.py in the evaluation folder. In theory, the script can be used for both evaluation of pre-processed files as well as processing and evalution of test sets on implemented models. Please note that some functionality of the script (e.g., automatic sectioning of input data into single-/double-talk sections with separate metrics) requires meta-data created from the test set generation macro script.

Examples for the evaluation calls are provided in the batch_script.py file. The following steps have to be taken to use the macro script:

1. Make sure all metric prerequisites are fulfilled.
2. Change the dir_path variable in GetaecTestSet.py according to your setup.
3. Generate test set / enhanced files: Using the below described dataset generation script is recommended. If test sets or enhanced files are already available, follow the example folder structure provided. Add the dataset definition to GetaecTestSet if necessary.
4. If you want to run inference on a new model, add it as described below.

#### Script Parameters

The following arguments can be set in the macro script:

```python
--model path:       "path to NN model (if implemented)"
--echoless:         "remove echo from test files"
--noiseless:        "remove noise from test files"
--noNEAR:           "remove NE speaker from test files"
--noaudio:          "disable writing of enhanced files in inference"
--size:             "Full/Partial: wether to evaluate all or just a subset of test files"
--dataset:          "handle of the test set (see GetaecTestSet.py)"
--model_select:     "handle of the model (defined in macro_script)"
--evalAECMOS:       "enable AECMOS evaluation"
--ERLEoverTime:     "enable EoT evaluation on the entire file"

--sampling_in:      "test data sampling rate"
--sampling_AEC:     "AEC inference sampling rate"

--cold_start:       "discard test set convergence splits"
--operation_mode:   "both/inference/evaluation"

--add_delay:        "add delay to echo component"
--SER_adjust:       "list of relative SER adjustments"
--SNR_adjust:       "list of relative SNR adjustments"
```

#### Adding new inference models

The macro script can easily expanded with new models for inference and evaluation. Two steps are required:

1. Adding the model initialization to the model choice (line 69): 
```python
elif Model_select == '<new_model_name>':
    your_model = [...]
```
2. Adding the model inference function to the macro script (line 325): 
```python
elif Model_select == '<new_model_name>':
    s_post = your_model(y,x,[...])
```


## 3. Dataset Generation

This toolbox also allows the generation of datasets featuring a high variety of conditions. In the following sections, we describe the available functions within the macro script. Currently, the dataset generation is implemented in Matlab, and allows only limited access to individual functions outside the macro script. In the future, this will be replaced by a pythonic and more modular implementation.

Note that we will mainly focus on describing the merit for generating test sets in the following sections. However, The same script can be used to create training and validation datasets, and offers a variety of functionality specifically for this purpose. The following steps need to be taken to generate data via the macro script:

1. Download the respective datasets (or define paths to existing ones).
2. Install [the actlevel function](https://github.com/foss-for-synopsys-dwc-arc-processors/G722/tree/master/sv56) by downloading the linked folder and compiling (C compiler required):
```C
cc -o actlevel actlevel.c sv-p56.c ugst-utl.c -lm
```
3. Adjust [params.m] file to reflect the desired condition.
4. If necessary, adjust file discovery in macro script to fit unfamiliar datasets.

### 3.1 Dataset Download

This section describes the retrieval of the (mostly freely available) datasets set up for the current macro script. Other datasets can be used, but might require manual implementation in the main.m file of the data generation script. Datasets are usually stored in the '00_Data' folder.

#### Speech Data

[CSTR-VCTK](https://datashare.ed.ac.uk/handle/10283/3443):
High quality recordings of English speakers with various accents. Provided in 48 kHz, requires downsampling to 16 kHz first.

[TIMIT](https://catalog.ldc.upenn.edu/LDC93S1): Common dataset in speech enhancement. Application fee required.

#### Noise Databases

[DEMAND](https://dcase-repo.github.io/dcase_datalist/datasets/scenes/demand.html): Provided in 48 kHz, downsampling to 16 kHz required.

[ETSI](https://docbox.etsi.org/stq/Open/EG%20202%20396-1%20Background%20noise%20database)

#### RIR Databases

The provided .mat files of the RIR databases are provided via Git LFS. To download them, make sure git LFS is installed:

```bash
git lfs install
```

[Aachen Impulse Response Database](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/): Collection of recorded real-world RIRs. Processed collection is provided in the AIR_binaural.mat file as part of this toolbox in accordance to the the provided MIT license.

Braunschweig Dynamic Impulse Response Database: Recordings of continuously changig RIR environment as described in  [[Jung, 2011]](https://ieeexplore.ieee.org/document/6639252). Files are provided as part of this toolbox.

### 3.2 Test Conditions

By adjusting the configuration of the macro script, the generated dataset can be adjusted to exhibit different conditions a system might encounter in practical applications. The generation of diverse testing conditions allows for a more in-depth evaluation of echo control systems, giving insight into their behaviour and potential shortcomings.

Test conditions can be adjusted via the parameter.m script. 

#### Double- and Single-Talk Conditions

The macro script allows for the explicit generation of DT and ST conditions, including convergence periods (with tracked section lengths to be used in the evaluation macro script).

Specifically, the DT condition will create a preceeding STFE and STNE section, while the ST conditions will add a preceding section of their respective type. Sections will be automatically seperated and individually evaluated in the evaluation macro script. If you run inference using the macro script, the "cold start" argument -CS can disable preceding sections for evaluation of unconverged performance.

```Matlab
param.test_condition = 1;           % Testing condition (with convergence periods)
                                    % 0 - off; 1 - DT; 2- STFE; 3 - STNE
param.len_sec = 8;                  % Condition section lengths
```

Choose test_condition = 0 for normal generation without preceeding convergence sections (you still can simulate STFE/STNE by disabling components in the evaluation macro script). The length of the sections can be controlled via the len_sec parameter. An example application can be found in [our journal](https://ieeexplore.ieee.org/document/10535257) in Section IV.A/B.

#### Far-End Excitation and Nonlinearity

The creation of various conditions for the loadspeaker signal can aid in training of robust neural network-based models and also help identify issues of existing methods with regards to unseen conditions.

For FE excitation, you can choose between speech and white Gaussian noise with the parameter WGNasFE. The parameter enableFENoise lets you add noise to an existing FE reference signal.

```Matlab
param.WGNasFE       = 1;    % replaces FE speaker with WGN
param.enableFENoise = 0;    % adds noise to FE signal (settings below)
```

You can also choose to apply a non-linear distortion to your loudspeaker signal, which is an important aspect of EC system evaluation, especially for classical, linear echo cancellers who might struggle in such regard. The toolbox offers the following implemented options:

- [Scaled error function](https://www.sciencedirect.com/science/article/abs/pii/S0893608021001258): adjustable parameter to control harshness
- [Memoryless Sigmoidal](https://web.cse.ohio-state.edu/~wang.77/papers/Zhang-Wang.interspeech18.pdf):  harsh nonlinearity with asymmetric mapping function
- [Arctan](https://ieeexplore.ieee.org/document/6639252): Simple arctan nonlinearity, rather mild

```Matlab
param.nonlinear     = 1;                        % enable LS non-linearities
param.NLfunc{1}     = 'SEF';                    % SEF       - scaled error function
% param.NLfunc{2}     = 'memSeg';               % memSeg    - memoryless sigmoidal after Wang
% param.NLfunc{3}     = 'arctan';               % arctan    - arctan after Jung
% param.NLfunc        = param.NLfunc';          % if multiple options passed (random choice per file)

param.NLoptions     = [0.5, 1, 10, 999];        % beta parameter for SEF
```

The parameter NLoptions controls the beta value of the SEF function (see [our journal article](https://ieeexplore.ieee.org/document/10535257), Section III.A, for details). An example application can be found in our journal in Section IV.E.

#### Echo Generation

The generation of diverse echo conditions is crucial, not only for training of neural network-based models, but also for an exhaustive evaluation of any EC algorithm. 

The (nonlinearily distorted) loudspeaker signal will be convolved with an impulse response, for which we provide several options, either real-world recordings or synthetically generated. The options are (given that prerequisites are fulfilled):

- [image](https://www.audiolabs-erlangen.de/fau/professor/habets/software/rir-generator): synthetically created RIR after image method; can be chosen from precomputed options or generated online pre file
- [exp](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2014/HTML/papers/1569912189.pdf): WGN with exponential decay, good for learning generalized RIR modelling
- [AIR](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/): Aachen Impulse Response Database; collection of recorded real-world RIRs
- [TUBS_dynIR](https://ieeexplore.ieee.org/document/6639252): continuously changing RIR from real-world recordings; applies sample-by-sample, which might take more time than other options

```Matlab
param.RIR_mode = 'exp'; % RIR mode used: 'imagePRE'(image method, precalculated)
                        % 'image' (image method, online calulation - slower),
                        % 'exp' (WGN with exponential decay),
                        % 'AIR' (Aachen IR DB)
                        % 'AIR_DC' (delay-compensated AIR)
                        % 'TUBS_dynIR' (TUBS continuously changing RIR after Jung)

param.generate_shortIR  = 512;
% if > 0, generates an additional echo scenario with shortened RIR
```

Specific methods might require additional method-specific parameters found in the param.m file.

If you wish to introduce a RIR change into your dataset, you can adjust the following parameters of the macro script:

```Matlab
param.IR_pathChange = [0,0];         % time frame ([start,stop] in s) in which RIR changes may happen
param.IR_fade       = [0.00, 0.00];  % fade-in time of new RIR ([min,max] in s)
```

Note that the script will automatically provide a version of the dataset without RIR switch for accurate analysis of reconvergence behaviour. An example application can be found in [our journal](https://ieeexplore.ieee.org/document/10535257) in Section IV.C.

[Technical comment] The RIR after the switch is used for generating the respective non-switch files. Therefore, both files share the same echo component after the RIR switch, allowing for accurate comparison of performance.

#### Continuously Changing Room Impulse Response

A special case of impulse response rarely reported is the continuously changing RIR. It poses a particularly difficult condition for EC models, as it requires constant reconvergence. We use recordings of [[Jung, 2011]](https://ieeexplore.ieee.org/document/6639252), whose setup is also reflected in [ITU-T P.1110](https://www.itu.int/rec/T-REC-P.1110/en) and [P.1130](https://www.itu.int/rec/T-REC-P.1130/en). A visualization of the recording setup is given below:

![BSdIR recording setup](Images/BSdIR_recording.PNG)

The image shows our car setup for dynamic RIR generation, with a handsfree microphone at rear-view mirror position and four loudspeakers. Changes in the RIR are simulated by rotating a reflecting surface on the passenger seat along the azimuth angle Φ.

The provided method for loading an RIR from the recordings' 
.mat files can be adjusted with the following parameters:

```Matlab
%% TUBS dynIR-specific values
param.dyn_length = 4;   % length of accessed recording in s; [1, 4, 8, 20]
param.fin_length = 8;   % length of generated RIR in s
param.step       = 1;   % step size (skip x-1 samples in recording)
param.freeze     = 4;   % time in s after which RIR is frozen
```

Note that this RIR, as it contains an individual vector for each recorded sample, requires considerable memory resources and time during dataset generation. An example application can be found in [our journal]() in Section IV.D.

#### SER/SNR adjustment

The macro script allows adjusting audio levels using [the ITU-P P.56 standard](https://github.com/foss-for-synopsys-dwc-arc-processors/G722/tree/master/sv56), which is mainly used for setting signal-to-echo and signal-to-noise ratios (SER/SNR). Download the linked folder and compile using the command (requires c compiler, e.g., gcc):

```C
cc -o actlevel actlevel.c sv-p56.c ugst-utl.c -lm
```

Place the actlevel.exe into the base 01_Preprocessing folder. When setting SER and SNR values, you can choose to pass a list, from which the respective value is picked randomly for each file:

```Matlab
param.dBov     = -21;               % base audio level (in dB)
param.refLevel = [-36, -21];        % reference sound level range (in dB)
...
param.SER  = [-20,-15,-10,-5,0,5,10];     % randomly chosen from list
...
param.SNR  = [0, 5, 10, 15, 20];          % randomly chosen from list
```

Passing 99 as a SER/SNR value will disable the respective component. If you want to control the SER/SNR during training or run evaluation on multiple specific SER/SNR conditions, it is recommended to set the values to 0 during dataset generation. The evaluation macro script contains arguments for relative(!) SER/SNR adjustment. An example application can be found in [our journal](https://ieeexplore.ieee.org/document/10535257) in Section IV.F.

#### Delay 

Delay is handled as a separate option in the evaluation macro script for fixed delay conditions. Note that you can remove original delay of the Aachen RIR database by using the option:

```Matlab
param.RIR_mode = 'AIR_DC';
```

## 4. Citation

If you use our toolbox for your research, please cite our work:

Application of the toolbox for AEC/AES analysis (will be updated once published):
```BibTex
@Article{Seidel2024,
  author      = {Ernst Seidel and Tim Fingscheidt},
  title       = {{Convergence and Performance Analysis of Classical, Hybrid, and Deep Acoustic Echo Control}},
  journal	= {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year		= {2024},
  volume	= {},
  number	= {},
  pages		= {},
  month		= jun,
}
```

Black-box metrics:
```BibTex
@InProceedings{Fingscheidt2007,
  author	= {T. Fing\-scheidt and S. Suhadi},
  booktitle	= {Proc. of Interspeech},
  title		= {{Quality Assessment of Speech Enhancement Systems by Separation of Enhanced Speech, Noise, and Echo}},
  year		= {2007},
  address	= {Antwerp, Belgium},
  month		= aug,
  pages		= {818--821},
  keywords	= {Signal Processing, Measure},
}

@InProceedings{Fingscheidt2008,
  author	= {Tim Fingscheidt and Suhadi Suhadi and K. Steinert},
  title		= {{Towards Objective Quality Assessment of Speech Enhancement Systems in a Black Box Approach}},
  booktitle	= {Proc. of ICASSP},
  year		= {2008},
  pages		= {273--276},
  address	= {Las Vegas, NV, USA},
  month		= apr,
}
```

Continuously Changing Impulse Response:
```BibTex
@InProceedings{Jung2013,
  author      = {Marc-André Jung and Lucca Richter and Tim Fingscheidt},
  title       = {{Towards Reproducible Evaluation of Automotive Hands-Free Systems in Dynamic Conditions}},
  booktitle   = {Proc. of ICASSP},
  year        = {2013},
  pages       = {8144--8148},
  address     = {Vancouver, Canada},
  month       = may,
}
```
