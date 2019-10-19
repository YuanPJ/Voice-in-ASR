import os
import re
import pickle
import shutil
import random
import argparse
import numpy as np
import librosa
from glob import glob
from os.path import join
from pathlib import Path


def main():
    # Make output directory
    output_dir = 'wav'
    os.makedirs(output_dir, exist_ok=True)

    # Make experiment
    exp = Path('./txt')
    exp = sorted(list(exp.rglob("*.txt")))
    for i, path in enumerate(exp):
        print(i, path)
        exp[i] = np.loadtxt(path, dtype=str, delimiter=' ')

    # Make sheet
    for eid, one_exp in enumerate(exp):
        sheet = make_sheet(eid+1, one_exp)
        with open(join('./', 'exp-%d.html' % (eid+1)), 'w') as f:
            f.write(sheet)


def make_sheet(exp_id, one_exp):
    html = '<html>\n\
            <head>\n\
                <meta charset="UTF-8">\
                <title>Human Verification Test - {}</title>\
                <link rel="stylesheet" type="text/css" href="./stylesheet.css"/>\n\
            </head>\n\
            <h1>Exp - {}</h1>\n\
            <table>\n'.format(exp_id, exp_id)
    for uid, uttr in enumerate(one_exp):
        # Add text
        txt_row = '<tr>\n'
        txt_row += '<td class="txt_fix_width" style="text-align: left" colspan="6">\n'
        txt_row += '{}.\n'.format(uid+1)
        txt_row += '</td>\n'
        txt_row += '</tr>\n'

        idx_row = '<tr>\n'
        audio_row = '<tr>\n'
        p0, p1, p2, ans = uttr

        for pid, p in enumerate([p0, p1, p2]):
            path = check(p, exp_id)
            idx_row += '<td style="text-align: center">\n'
            if pid == 0:
                idx_row += '{}-A'.format(uid+1)
            elif pid == 1:
                idx_row += '{}-B'.format(uid+1)
            else:
                idx_row += '{}-?'.format(uid+1)
            idx_row += '</td>\n'

            audio_row += '<td style="text-align: center">\n'
            audio_row += '<audio controls> <source src="{}" type="audio/wav"></audio>\n'.format(
                path)
            audio_row += '</td>\n'

        idx_row += '</tr>\n'
        audio_row += '</tr>\n'
        audio_row += '<tr style="height: 40px;"> <!-- Mimic the margin --></tr>\n'

        html += txt_row + idx_row + audio_row
    html += '</table>'
    return html


def check(path, exp_id):
    p = Path(path)
    if not p.exists():
        raise NotImplementedError

    wav_path = 'wav/' + str(exp_id) + '/' + p.name.replace('.flac', '.wav')
    if not Path(wav_path).exists():
        wav, sr = librosa.load(p, sr=16000, mono=True)
        librosa.output.write_wav(wav_path, wav, sr=16000)
    return wav_path


if __name__ == '__main__':
    main()
