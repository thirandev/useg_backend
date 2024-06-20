import numpy as np
import scipy.signal
from tqdm import tqdm
import gc


class ImageBlender:
    def __init__(self, window_size, subdivisions, nb_classes, pred_func):
        self.window_size = window_size
        self.subdivisions = subdivisions
        self.nb_classes = nb_classes
        self.pred_func = pred_func

    def _spline_window(self, window_size, power=2):
        intersection = int(window_size / 4)
        wind_outer = (abs(2 * (scipy.signal.triang(window_size))) ** power) / 2
        wind_outer[intersection:-intersection] = 0

        wind_inner = 1 - (abs(2 * (scipy.signal.triang(window_size) - 1)) ** power) / 2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0

        wind = wind_inner + wind_outer
        wind = wind / np.average(wind)
        return wind

    def _window_2D(self, window_size, power=2):
        key = "{}_{}".format(window_size, power)
        if key in self.cached_2d_windows:
            wind = self.cached_2d_windows[key]
        else:
            wind = self._spline_window(window_size, power)
            wind = np.expand_dims(np.expand_dims(wind, 1), 1)
            wind = wind * wind.transpose(1, 0, 2)
            self.cached_2d_windows[key] = wind
        return wind

    def _pad_img(self, img, window_size, subdivisions):
        aug = int(round(window_size * (1 - 1.0 / subdivisions)))
        more_borders = ((aug, aug), (aug, aug), (0, 0))
        ret = np.pad(img, pad_width=more_borders, mode='reflect')
        return ret

    def _unpad_img(self, padded_img, window_size, subdivisions):
        aug = int(round(window_size * (1 - 1.0 / subdivisions)))
        ret = padded_img[aug:-aug, aug:-aug, :]
        return ret

    def _rotate_mirror_do(self, im):
        mirrs = []
        mirrs.append(np.array(im))
        mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
        mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
        mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
        im = np.array(im)[:, ::-1]
        mirrs.append(np.array(im))
        mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
        mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
        mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
        return mirrs

    def _rotate_mirror_undo(self, im_mirrs):
        origs = []
        origs.append(np.array(im_mirrs[0]))
        origs.append(np.rot90(np.array(im_mirrs[1]), axes=(0, 1), k=3))
        origs.append(np.rot90(np.array(im_mirrs[2]), axes=(0, 1), k=2))
        origs.append(np.rot90(np.array(im_mirrs[3]), axes=(0, 1), k=1))
        origs.append(np.array(im_mirrs[4])[:, ::-1])
        origs.append(np.rot90(np.array(im_mirrs[5]), axes=(0, 1), k=3)[:, ::-1])
        origs.append(np.rot90(np.array(im_mirrs[6]), axes=(0, 1), k=2)[:, ::-1])
        origs.append(np.rot90(np.array(im_mirrs[7]), axes=(0, 1), k=1)[:, ::-1])
        return np.mean(origs, axis=0)

    def _windowed_subdivs(self, padded_img, window_size, subdivisions, nb_classes, pred_func):
        WINDOW_SPLINE_2D = self._window_2D(window_size=window_size, power=2)
        step = int(window_size / subdivisions)
        padx_len = padded_img.shape[0]
        pady_len = padded_img.shape[1]
        subdivs = []

        for i in range(0, padx_len - window_size + 1, step):
            subdivs.append([])
            for j in range(0, pady_len - window_size + 1, step):
                patch = padded_img[i:i + window_size, j:j + window_size, :]
                subdivs[-1].append(patch)

        gc.collect()
        subdivs = np.array(subdivs)
        gc.collect()
        a, b, c, d, e = subdivs.shape
        subdivs = subdivs.reshape(a * b, c, d, e)
        gc.collect()

        subdivs = pred_func(subdivs)
        gc.collect()
        subdivs = np.array([patch * WINDOW_SPLINE_2D for patch in subdivs])
        gc.collect()

        subdivs = subdivs.reshape(a, b, c, d, nb_classes)
        gc.collect()

        return subdivs

    def _recreate_from_subdivs(self, subdivs, window_size, subdivisions, padded_out_shape):
        step = int(window_size / subdivisions)
        padx_len = padded_out_shape[0]
        pady_len = padded_out_shape[1]
        y = np.zeros(padded_out_shape)
        a = 0
        for i in range(0, padx_len - window_size + 1, step):
            b = 0
            for j in range(0, pady_len - window_size + 1, step):
                windowed_patch = subdivs[a, b]
                y[i:i + window_size, j:j + window_size] = y[i:i + window_size, j:j + window_size] + windowed_patch
                b += 1
            a += 1
        return y / (subdivisions ** 2)

    def predict_img_with_smooth_windowing(self, input_img):
        pad = self._pad_img(input_img, self.window_size, self.subdivisions)
        pads = self._rotate_mirror_do(pad)

        res = []
        for pad in tqdm(pads):
            sd = self._windowed_subdivs(pad, self.window_size, self.subdivisions, self.nb_classes, self.pred_func)
            one_padded_result = self._recreate_from_subdivs(sd, self.window_size, self.subdivisions,
                                                            padded_out_shape=list(pad.shape[:-1]) + [self.nb_classes])
            res.append(one_padded_result)

        padded_results = self._rotate_mirror_undo(res)
        prd = self._unpad_img(padded_results, self.window_size, self.subdivisions)
        prd = prd[:input_img.shape[0], :input_img.shape[1], :]

        return prd
