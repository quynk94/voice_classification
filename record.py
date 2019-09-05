#!/usr/bin/env python3

import argparse
import tempfile
from queue import Queue
import time
from datetime import datetime as dt


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-l', '--list-devices', action='store_true',
                    help='show list of audio devices and exit')
parser.add_argument('-d', '--device', type=int_or_str,
                    help='input device (numeric ID or substring)')
parser.add_argument('-r', '--samplerate', type=int, help='sampling rate')
parser.add_argument('-c', '--channels', type=int, default=1,
                    help='number of input channels')
parser.add_argument('filename', nargs='?', metavar='FILENAME',
                    help='audio file to store recording to')
parser.add_argument('-t', '--subtype', type=str,
                    help='sound file subtype (e.g. "PCM_24")')
parser.add_argument('-s', '--split-after', type=int, default=10,
                    help='split after this number of seconds')
args = parser.parse_args()

try:
    import sounddevice as sd
    import soundfile as sf

    if args.list_devices:
        print(sd.query_devices())
        parser.exit()
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        # soundfile expects an int, sounddevice provides a float:
        args.samplerate = int(device_info['default_samplerate'])
    queue = Queue()

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, flush=True)
        queue.put(indata.copy())

    with sd.InputStream(samplerate=args.samplerate, device=args.device,
                        channels=args.channels, callback=callback):
        counter = 0
        start_time = time.time()
        while True:
            now = dt.now()
            context = {'counter': counter, 'year': now.year, 'month': now.month,
                       'day': now.day, 'hour': now.hour, 'minute': now.minute, 'second': now.second}
            if args.filename is None:
                filename = tempfile.mktemp(prefix='rec_unlimited_%d_' % counter,
                                           suffix='.wav', dir='')
            else:
                filename = args.filename.format(**context)
            with sf.SoundFile(filename, mode='x', samplerate=args.samplerate,
                              channels=args.channels, subtype=args.subtype) as file:
                print("Now writing to: " + repr(filename))
                while True:
                    if time.time() - start_time > args.split_after:
                        start_time += args.split_after
                        counter += 1
                        break
                    file.write(queue.get())

except KeyboardInterrupt:
    parser.exit('\nRecording finished.')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
