import numpy as np
import librosa


def generate(samples, sample_rate):
    # Change pitch and speed
    y_pitch_speed = samples.copy()
    # you can change low and high here
    length_change = np.random.uniform(low=0.8, high=1)
    speed_fac = 1.0 / length_change
    print("resample length_change = ", length_change)
    tmp = np.interp(np.arange(0, len(y_pitch_speed), speed_fac),
                    np.arange(0, len(y_pitch_speed)), y_pitch_speed)
    minlen = min(y_pitch_speed.shape[0], tmp.shape[0])
    y_pitch_speed *= 0
    y_pitch_speed[0:minlen] = tmp[0:minlen]

    # Change pitch only
    y_pitch = samples.copy()
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2*(np.random.uniform())
    print("pitch_change = ", pitch_change)
    y_pitch = librosa.effects.pitch_shift(y_pitch.astype('float64'),
                                          sample_rate, n_steps=pitch_change,
                                          bins_per_octave=bins_per_octave)

    # Change speed only
    y_speed = samples.copy()
    speed_change = np.random.uniform(low=0.9, high=1.1)
    print("speed_change = ", speed_change)
    tmp = librosa.effects.time_stretch(y_speed.astype('float64'), speed_change)
    minlen = min(y_speed.shape[0], tmp.shape[0])
    y_speed *= 0
    y_speed[0:minlen] = tmp[0:minlen]

    # value augmentation
    y_aug = samples.copy()
    dyn_change = np.random.uniform(low=1.5, high=3)
    print("dyn_change = ", dyn_change)
    y_aug = y_aug * dyn_change
    print(y_aug[:50])
    print(samples[:50])

    # add distribution noise
    y_noise = samples.copy()
    # you can take any distribution from https://docs..org/doc/numpy-1.13.0/reference/routines.random.html
    noise_amp = 0.005*np.random.uniform()*np.amax(y_noise)
    y_noise = y_noise.astype('float64') + noise_amp * \
        np.random.normal(size=y_noise.shape[0])

    # random shifting
    y_shift = samples.copy()
    timeshift_fac = 0.2 * 2*(np.random.uniform()-0.5)  # up to 20% of length
    print("timeshift_fac = ", timeshift_fac)
    start = int(y_shift.shape[0] * timeshift_fac)
    print(start)
    if (start > 0):
        y_shift = np.pad(y_shift, (start, 0), mode='constant')[
            0:y_shift.shape[0]]
    else:
        y_shift = np.pad(y_shift, (0, -start),
                         mode='constant')[0:y_shift.shape[0]]

    # apply hpss
    y_hpss = librosa.effects.hpss(samples.astype('float64'))
    print(y_hpss[1][:10])
    print(samples[:10])

    # Shift silent to the right
    sampling = samples[(samples > 200) | (samples < -200)]
    shifted_silent = sampling.tolist(
    )+np.zeros((samples.shape[0]-sampling.shape[0])).tolist()

    # Streching
    input_length = len(samples)
    streching = samples.copy()
    streching = librosa.effects.time_stretch(streching.astype('float'), 1.1)
    if len(streching) > input_length:
        streching = streching[:input_length]
    else:
        streching = np.pad(
            streching, (0, max(0, input_length - len(streching))), "constant")

    return [
        y_pitch_speed,
        y_pitch,
        y_speed,
        y_aug,
        y_noise,
        y_shift,
        y_hpss,
        streching
    ]
