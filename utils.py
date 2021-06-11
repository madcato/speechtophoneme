"""
Defines various functions for processing the data.
"""
import numpy as np
import soundfile
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from numpy.lib.stride_tricks import as_strided
from char_map import char_map, index_map, get_number_of_char_classes
# import soundfile as sf

num_classes = get_number_of_char_classes()

def featurize_mfcc(audio_clip, mfcc_dim=26, lowfreq=0, highfreq=8000, winlen=0.025, winstep=0.01):
    """ For a given audio clip, calculate the corresponding mfcc
    Params:
        audio_clip (str): Path to the audio clip
    """
    #######
    (rate, sig) = wav.read(audio_clip)
#     if sig.dtype == 'int16':
#         nb_bits = 16 # -> 16-bit wav files
#     elif sig.dtype == 'int32':
#         nb_bits = 32 # -> 32-bit wav files
#     max_nb_bit = float(2 ** (nb_bits - 1))
#sig = sig / (max_nb_bit + 1.0) # sam
    #####
    # sig, rate = sf.read(audio_clip)
    #####
    #####
    # print("wavshape: ")
    # print(sig.shape)
    # print(sig.dtype)
    # print(sig[1024])
    # print(sig[11024])
    # print(sig[0])
    # print(sig[1])
    # return mfcc(sig, rate, numcep=mfcc_dim, lowfreq=lowfreq, highfreq=highfreq)# winlen=0.025,winstep=0.01
    return mfcc(sig,samplerate=rate,winlen=winlen,winstep=winstep,numcep=mfcc_dim,
                     nfilt=mfcc_dim,nfft=512,lowfreq=lowfreq,highfreq=highfreq,preemph=0.97,
         ceplifter=22,appendEnergy=True)

def featurize_spectogram(audio_clip, step=10, window=20, max_freq=8000):
    """ For a given audio clip, calculate the corresponding spectogram
    Params:
        audio_clip (str): Path to the audio clip
    """
    return spectrogram_from_file(
        audio_clip, step=step, window=window,
        max_freq=max_freq)

def calc_feat_dim(window, max_freq):
    return int(0.001 * window * max_freq) + 1

def conv_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


def spectrogram(samples, fft_length=256, sample_rate=2, hop_length=128):
    """
    Compute the spectrogram for a real signal.
    The parameters follow the naming convention of
    matplotlib.mlab.specgram

    Args:
        samples (1D array): input audio signal
        fft_length (int): number of elements in fft window
        sample_rate (scalar): sample rate
        hop_length (int): hop length (relative offset between neighboring
            fft windows).

    Returns:
        x (2D array): spectrogram [frequency x time]
        freq (1D array): frequency of each row in x

    Note:
        This is a truncating computation e.g. if fft_length=10,
        hop_length=5 and the signal has 23 elements, then the
        last 3 elements will be truncated.
    """
    assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window**2)

    # The scaling below follows the convention of
    # matplotlib.mlab.specgram which is the same as
    # matlabs specgram.
    scale = window_norm * sample_rate

    trunc = (len(samples) - fft_length) % hop_length
    x = samples[:len(samples) - trunc]

    # "stride trick" reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)

    # window stride sanity check
    assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

    # broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x)**2

    # scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale

    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

    return x, freqs


def spectrogram_from_file(filename, step=10, window=20, max_freq=None,
                          eps=1e-14):
    """ Calculate the log of linear spectrogram from FFT energy
    Params:
        filename (str): Path to the audio file
        step (int): Step size in milliseconds between windows
        window (int): FFT window size in milliseconds
        max_freq (int): Only FFT bins corresponding to frequencies between
            [0, max_freq] are returned
        eps (float): Small value to ensure numerical stability (for ln(x))
    """
    with soundfile.SoundFile(filename) as sound_file:
        audio = sound_file.read(dtype='float32')
        sample_rate = sound_file.samplerate
        if audio.ndim >= 2:
            audio = np.mean(audio, 1)
        if max_freq is None:
            max_freq = sample_rate / 2
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must not be greater than half of "
                             " sample rate")
        if step > window:
            raise ValueError("step size must not be greater than window size")
        hop_length = int(0.001 * step * sample_rate)
        fft_length = int(0.001 * window * sample_rate)
        pxx, freqs = spectrogram(
            audio, fft_length=fft_length, sample_rate=sample_rate,
            hop_length=hop_length)
        ind = np.where(freqs <= max_freq)[0][-1] + 1
    return np.transpose(np.log(pxx[:ind, :] + eps))

def text_to_int_sequence(text):
    """ Convert text to an integer sequence """
    int_sequence = []
    for c in text:
        ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence

def phoneme_to_int(phoneme):
    return char_map[phoneme]

def phoneme_to_int_sequence(text):
    """ Convert text to an integer sequence """
    # TODO: check phoneme duration
    int_sequence = []
    i = 0
    length = len(text)
    while i < length:
        # check double symbol phoneme
        phoneme = text[i]
        ch = -1
        if i + 1 < length:
            phoneme += text[i+1]
            if phoneme in char_map.keys(): 
                ch = char_map[phoneme]
                i += 1
        if ch == -1:
            phoneme = text[i]
            if phoneme in char_map.keys():
                ch = char_map[phoneme]
        i += 1
        if ch == -1:
            print(f"Invalid phoneme {phoneme} for word: {text}")
        int_sequence.append(ch)
    return int_sequence

def int_sequence_to_text(int_sequence, add=0):
    """ Convert an integer sequence to text """
    text = []
    for a in int_sequence:
        c = int(a) + add
        if c == 66:
            ch = ''
        else:
            ch = index_map[c]
        text.append(ch)
    return text

def load_model_checkpoint(path, summary=True):

    #this is a terrible hack
    from keras.utils.generic_utils import get_custom_objects
    # get_custom_objects().update({"tf": tf})
    get_custom_objects().update({"clipped_relu": clipped_relu})
    get_custom_objects().update({"selu": selu})
    # get_custom_objects().update({"TF_NewStatus": None})

    jsonfilename = path+".json"
    weightsfilename = path+".h5"

    json_file = open(jsonfilename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    K.set_learning_phase(1)
    loaded_model = model_from_json(loaded_model_json)

    # load weights into loaded model
    loaded_model.load_weights(weightsfilename)
    # loaded_model = load_model(path, custom_objects=custom_objects)


    if(summary):
        loaded_model.summary()

    return loaded_model

def load_cmodel_checkpoint(path, summary=True):

    #this is a terrible hack
    from keras.utils.generic_utils import get_custom_objects
    # get_custom_objects().update({"tf": tf})
    get_custom_objects().update({"clipped_relu": clipped_relu})
    get_custom_objects().update({"selu": selu})
    # get_custom_objects().update({"TF_NewStatus": None})

    cfilename = path+".h5"

    K.set_learning_phase(1)
    loaded_model = load_model(cfilename)


    if(summary):
        loaded_model.summary()

    return loaded_model

