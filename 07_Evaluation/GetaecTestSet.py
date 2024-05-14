def GetaecTestSet(DatasetType, Model_select):

    ###########################
    # Path Setup
    ###########################

    dir_path = '..'      # enter your data root folder path here / relative path works as well

    signalPaths = dict()
    BB_avail = 0            # signal components (e.g., s/d/n) are not available
    sectioned = 0           # no sectioning; metrics computed over full file

    EnhancementTag = '_Enhanced'

    if DatasetType == 'Test_Set':
        FilePathBase = dir_path + '/02_Speech_Data/Example_files/test'  # base path
        signalPaths['y'] = FilePathBase + '/nearend_mic'                # standard saving folder
        signalPaths['x'] = FilePathBase + '/farend_speech'
        signalPaths['d'] = FilePathBase + '/echo'
        signalPaths['s'] = FilePathBase + '/nearend_speech'
        signalPaths['n'] = FilePathBase + '/nearend_noise'
        signalPaths['meta'] = FilePathBase + '/meta'
        BB_avail = 1                                                    # signal components (e.g., s/d/n) are available
        sectioned = 1                                                   # DT condition sectioning

    # examples from Microsoft AEC Challenge
    elif DatasetType == 'bICASSP_DT':
        FilePathBase = dir_path + '/02_Speech_Data/Microsoft/blind_test_set_icassp2022/doubletalk'
    elif DatasetType == 'bICASSP_STFE':
        FilePathBase = dir_path + '/02_Speech_Data/Microsoft/blind_test_set_icassp2022/farend-singletalk'
    elif DatasetType == 'bICASSP_STNE':
        FilePathBase = dir_path + '/02_Speech_Data/Microsoft/blind_test_set_icassp2022/nearend-singletalk'

    # examples test set definitions
    elif DatasetType == 'TUB_synth_test8_WGNrealDTch_16':
        FilePathBase = dir_path + '/02_Speech_Data/TIMIT/WGN8_unseenETSI_Jung_AIRch4_DT/files/test'
        signalPaths['y'] = FilePathBase + '/mic_ch1'            # saving folder for IR change condition
        signalPaths['x'] = FilePathBase + '/farend_speech'
        signalPaths['d'] = FilePathBase + '/echo_ch1'           # saving folder for IR change condition
        signalPaths['s'] = FilePathBase + '/nearend_speech'
        signalPaths['n'] = FilePathBase + '/nearend_noise'
        signalPaths['meta'] = FilePathBase + '/meta'
        BB_avail = 1                                            # signal components (e.g., s/d/n) are available
        sectioned = 1                                           # DT condition sectioning
    elif DatasetType == 'TUB_synth_test8_WGNrealFEch_16':
        FilePathBase = dir_path + '/02_Speech_Data/TIMIT/WGN8_unseenETSI_Jung_AIRch4_STFE/files/test'
        signalPaths['y'] = FilePathBase + '/mic_ch1'
        signalPaths['x'] = FilePathBase + '/farend_speech'
        signalPaths['d'] = FilePathBase + '/echo_ch1'
        signalPaths['s'] = FilePathBase + '/nearend_speech'
        signalPaths['n'] = FilePathBase + '/nearend_noise'
        signalPaths['meta'] = FilePathBase + '/meta'
        BB_avail = 1
        sectioned = 2                                           # STFE condition sectioning
    elif DatasetType == 'TUB_synth_test_VCTKdevDT_16':
        FilePathBase = dir_path + '/02_Speech_Data/CSTR-VCTK/VCTK_seenETSI_SEF_ImageFE_16_DT/files/test'
        signalPaths['y'] = FilePathBase + '/nearend_mic'        # standard saving folder
        signalPaths['x'] = FilePathBase + '/farend_speech'
        signalPaths['d'] = FilePathBase + '/echo'               # standard saving folder
        signalPaths['s'] = FilePathBase + '/nearend_speech'
        signalPaths['n'] = FilePathBase + '/nearend_noise'
        signalPaths['meta'] = FilePathBase + '/meta'
        BB_avail = 1
        sectioned = 1
    elif DatasetType == 'TUB_synth_test_VCTKdevFE_16':
        FilePathBase = dir_path + '/02_Speech_Data/CSTR-VCTK/VCTK_seenETSI_SEF_ImageFE_16_STFE/files/test'
        signalPaths['y'] = FilePathBase + '/nearend_mic'
        signalPaths['x'] = FilePathBase + '/farend_speech'
        signalPaths['d'] = FilePathBase + '/echo'
        signalPaths['s'] = FilePathBase + '/nearend_speech'
        signalPaths['n'] = FilePathBase + '/nearend_noise'
        signalPaths['meta'] = FilePathBase + '/meta'
        BB_avail = 1
        sectioned = 2
    elif DatasetType == 'TUB_synth_test_VCTKdevNE_16':
        FilePathBase = dir_path + '/02_Speech_Data/CSTR-VCTK/VCTK_seenETSI_SEF_ImageFE_16_STNE/files/test'
        signalPaths['y'] = FilePathBase + '/nearend_mic'
        signalPaths['x'] = FilePathBase + '/farend_speech'
        signalPaths['d'] = FilePathBase + '/echo'
        signalPaths['s'] = FilePathBase + '/nearend_speech'
        signalPaths['n'] = FilePathBase + '/nearend_noise'
        signalPaths['meta'] = FilePathBase + '/meta'
        BB_avail = 1
        sectioned = 3                                           # STNE condition sectioning
    elif DatasetType == 'TUB_synth_test_hard_16':
        FilePathBase = dir_path + '/02_Speech_Data/CSTR-VCTK/VCTK_DEMAND_Wang_ImageFE_16/files/test'
        signalPaths['y'] = FilePathBase + '/nearend_mic'
        signalPaths['x'] = FilePathBase + '/farend_speech'
        signalPaths['d'] = FilePathBase + '/echo'
        signalPaths['s'] = FilePathBase + '/nearend_speech'
        signalPaths['n'] = FilePathBase + '/nearend_noise'
        signalPaths['meta'] = FilePathBase + '/meta'
        BB_avail = 1
        sectioned = 0
        
    audio_out_path = dir_path + '/08_Results/' + Model_select + EnhancementTag + '/' + DatasetType
    signalPaths['shat'] = audio_out_path
 
    return FilePathBase, BB_avail, signalPaths, sectioned