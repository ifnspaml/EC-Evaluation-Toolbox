# Acoustic Echo Control Evaluation Toolbox



## 1. Introduction

Acoustic echo cancellation (AEC) and suppression (AES) are widely researched topics. However, only few papers about hybrid or deep acoustic echo control provide a solid comparative analysis of their methods as it was common with classical signal processing approaches. There can be distinct differences in the behaviour of an AEC/AES model which cannot be fully represented by a single metric or test condition, especially when comparing classical signal processing and machine-learned approaches. These characteristics include convergence behaviour, reliability under varying speech levels or far-end signal types, as well as robustness to adverse conditions such as harsh nonlinearities, room impulse response switches or continuous changes, or delayed echo. 
We provide a toolbox that allows evaluation on an extended set of test conditions and metrics, mainly focussed around the application on 16 kHz signals.

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

### 1.1 Roadmap

- [ ] GitHub Release
- [ ] Dataset provision / expanded DL aid
- [ ] expand dataset support
- [ ] expanded model zoo
- [ ] transition to unified framework for data generation, training, and testing

### 1.2 Prerequisites / Compatibility

Operating system:
All scripts were previously run on Windows 10 machines, and the evaluation scripts was also tested on CentOS.

Matlab:
The data generation in this toolbox was conducted on Matlab 2021b.

Python Requisites:

All packages are listed as used for the evaluation in our journal article. While not a definitive requirement, deviating from the reported versions might cause some functionality to break.

| Module    | version   | | Module    | version   |
| --------  | -------   |-| --------  | -------   |
| Python    | 3.12      | | numpy     | 1.26.4    |  
| PyTorch   | 2.2.0     | | scipy     | 1.12.0    |
| CUDA      | 11.8      | | soundfile | 0.12.1    |
|           |           | | librosa   | 0.10.1    |
|           |           | | webrtcvad | 2.0.10    |


## 2. Metrics (and Metric Setup Instructions)

As part of this toolbox, we provide various metrics, usable both as standalone python functions (on preprocessed files) or using the macro script for datasets. Some of the metrics require the download of third party source code as detailed below in order to function. All metrics can be accessed individually, or via the macro script (detailed below).

Note: All metrics can be verified by running the metric_unittest.py file in the Evaluation script folder.

### 2.1 Individual Metrics

#### Perceptual Evaluation of Speech Quality (PESQ)

The perceptual evalution of quality (PESQ) metric is a widely used metric for the assessment of speech quality, standardized as ITU-T recommendation P.862, and regularly used in evaluation of echo control mechanism. Please note that with the presence of unsuppressed background noise, results are expected to yield low scores.

[pesq git]

Once installed correctly, you can call PESQ through our macro script (details below) or by calling:

```python
from evaluation_metrics import compute_PESQ

result = compute_PESQ(s, e, sampling rate, int_range_flag=False)
```

#### Echo Return Loss Enhancement (ERLE)

The echo return loss enhancement (ERLE) metric is used to evaluate the reduction of the far-end echo. It can be accessed via 

```python
from evalution_metrics import compute_ERLE

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

# talk_type in ['nst', 'fst', 'dt']; refering to single-talk far-end, single-talk near-end, and double-talk
aecmos      = AECMOSEstimator(os.getcwd() +'/models/Run_1663915512_Stage_0.onnx')
echo, other = aecmos.run(talk_type='dt', lpb_sig=x,mic_sig=y,enh_sig=e)
```

Note that this function evaluates on the entire sequence and does not contain the trimming of test files present in the original source code. 

#### Log-Spectral Distance (LSD)

The log-spectral distance (LSD) metric,much like PESQ, reports overall quality (both NE speech preservation and echo suppression effectiveness affect the score), but is a straightforward distance metric. As such, it is less prone to performance differences getting masked by noise, but in return is less descriptive on the perceptual impact of residual echo and NE speech degradation. It can be accessed via:

```python
from evaluation_metrics import compute_LSD

LSD_score = compute_LSD(s, e)
```

### 2.2 Black-Box Metric Variants

PESQ, ERLE, and LSD are also available as black-box variants. The use of the black-box algorithm allows the disentanglement components in the enhanced signal, which can be used for a more precise evaluation of certain performance aspects (e.g., measuring NE PESQ without the disturbing influence of background noise). The black-box components can be computed and applied via:

```python
from audio_processing import get_BB_components
from evaluation_metrics import *

params = {'window': 'Blackman', 'fft_size': 512, 'window_shift': 64}

# components: NE speech [s], echo [d], noise [n]
[s_tilde, d_tilde, n_tilde] = get_BB_components(y, components=[s, d, n], params=params)

result                      = compute_PESQ(s, s_tilde, sampling rate, wb_flag=True, int_range_flag=False)
mean_erle, erle_over_time   = compute_ERLE(y, s_tilde, d, d_tilde, s_f=0.99)
LSD_score                   = compute_LSD(s, s_tilde)
```

Note that compute_erle takes an additional argument in black-box configuration.
While the black-box parameters are adjustable, it is recommended to leave them as is for optimal component disentanglement.

### 2.3 Evaluation Macro Script

The evaluation of all metrics over a large number of files can be automated by the provided macro script score_AEC.py in the evaluation folder. In theory, the script can be used for both evaluation of pre-processed files as well as processing and evalution of test sets on implemented models. Please note that some functionality of the script (e.g., automatic sectioning of input data into single-/double-talk sections with separate metrics) requires meta-data created from the test set generation macro script.

Examples for the evaluations conducted during our journal paper investigations are provided in the batch_script.py file. The following steps have to be taken to use the macro script:

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

Note that we will mainly focus on describing the merit for generating test sets in the following sections. However, The same script can be used to create training and validation datasets, and offers a variety of functionality specifically for this purpose. The folowing steps need to be taken to generate data via the macro script:

1. Download the respective datasets (or define paths to existing ones).
2. Adjust [params.m] file to reflect the desired condition.
3. If necessary, adjust file discovery in macro script to fit dataset. [...]

### 3.1 Dataset Download

This section describes the retrieval of the (mostly freely available) datasets set up for the current macro script. Other datasets can be used, but might require manual implementation into the data generation script (especially speech data due to separation of male and female speakers). Datasets are usually stored in the '00_Database' folder.

#### Speech Data

[CSTR-VCTK](https://datashare.ed.ac.uk/handle/10283/3443):
High quality recordings of English speakers with various accents. Provided in 48 kHz ,requires downsampling to 16 kHz first. [TODO: guide]

[TIMIT](https://catalog.ldc.upenn.edu/LDC93S1): Common dataset in speech enhancement. Application fee required.

#### Noise Databases

[DEMAND](https://dcase-repo.github.io/dcase_datalist/datasets/scenes/demand.html): Provided in 48 kHz, resampling to 16 kHz required.

[ETSI](https://docbox.etsi.org/stq/Open/EG%20202%20396-1%20Background%20noise%20database)

#### RIR Databases

Aachen Impulse Response Database:

Dynamic Impulse Response [Jung]:

### 3.2 Test Conditions

By adjusting the configuration of the macro script, the generated dataset can be adjusted to exhibit different conditions a system might encounter in practical applications. The generation of diverse testing conditions allows for a more in-depth evaluation of echo control systems, giving insight into their behaviour and potential shortcomings.

Test conditions can be adjusted via the [parameter.m] script. 

#### Double- and Single-Talk Conditions

The macro script allows for the explicit generation of DT and ST conditions, including convergence periods (with tracked section lengths to be used in the evaluation macro script).

Specifically, the DT condition will create a preceeding STFE and STNE section, while the ST conditions will add a preceding section of their respective type. Sections will be automatically seperated and individually evaluated in the evaluation macro script. If you run inference using the macro script, the "cold start" argument -CS can disable preceding sections for evaluation of unconverged performance.

```Matlab
param.test_condition = 1;           % Testing condition (with convergence periods)
                                    % 0 - off; 1 - DT; 2- STFE; 3 - STNE
param.len_sec = 8;                  % Condition section lengths
```

Choose test_condition = 0 for normal generation without preceeding convergence sections (you still can simulate STFE/STNE by disabling components in the evaluation macro script). The length of the sections can be controlled via the len_sec parameter.

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

The parameter NLoptions controls the beta value of the SEF function (see our journal article, Section III.A, for details).

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

Note that the script will automatically provide a version of the dataset without RIR switch for accurate analysis of reconvergence behaviour.

[Technical comment] The RIR after the switch is used for generating the respective non-switch files. Therefore, both files share the same echo component after the RIR switch, allowing for accurate comparison of performance.

#### Continuously Changing Room Impulse Response

A special case of impulse response rarely reported is the continuously changing RIR. It poses a particularly difficult condition for EC models, as it requires constant reconvergence. We use recordings of Jung [REF], whose setup is also reflected in ITU-T P.1110 [REF] and P.1130 [REF]. The provided method can be adjusted with the following parameters:

```Matlab
%% TUBS dynIR-specific values
param.dyn_length = 4;   % length of accessed recording in s; [1, 4, 8, 20]
param.fin_length = 8;   % length of generated RIR in s
param.step       = 1;   % step size (skip x-1 samples in recording)
param.freeze     = 4;   % time in s after which RIR is frozen
```

Note that this RIR, as it contains an individual vector for each recorded sample, requires considerable memory resources and time during dataset generation.

#### SER/SNR adjustment

The macro script allows adjusting audio levels using [the ITU-P P.56 standard](https://github.com/YouriT/matlab-speech/blob/master/MATLAB_CODE_SOURCE/voicebox/activlev.m), which is mainly used for setting signal-to-echo and signal-to-noise ratios (SER/SNR). When setting SER and SNR values, you can choose to pass a list, from which the respective value is picked randomly for each file:

```Matlab
param.dBov     = -21;               % base audio level (in dB)
param.refLevel = [-36, -21];        % reference sound level range (in dB)
...
param.SER  = [-20,-15,-10,-5,0,5,10];     % randomly chosen from list
...
param.SNR  = [0, 5, 10, 15, 20];          % randomly chosen from list
```

Passing 99 as a SER/SNR value will disable the respective component. If you want to control the SER/SNR during training or run evaluation on multiple specific SER/SNR conditions, it is recommended to set the values to 0 during dataset generation. The evaluation macro script contains arguments for relative(!) SER/SNR adjustment.

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
