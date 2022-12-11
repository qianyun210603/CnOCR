# coding: utf-8
# Copyright (C) 2021, [Breezedeus](https://github.com/breezedeus).
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
The previous version of this file is coded by my colleague Chuhao Chen.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from scipy.ndimage import shift


THRESHOLD = 145  # for white background
TABLE = [1]*THRESHOLD + [0]*(256-THRESHOLD)


def line_split(image, table=TABLE, split_threshold=0, blank=True, **_):
    """
    :param image: PIL.Image类型的原图或numpy.ndarray
    :param table: 二值化的分布值，默认值即可
    :param split_threshold: int, 分割阈值
    :param blank: bool,是否留白.True会保留上下方的空白部分
    :return: list,元素为按行切分出的子图与位置信息的list
    """
    if not isinstance(image, Image.Image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            raise TypeError

    image_ = image.convert('L')
    bn = image_.point(table, '1')
    bn_mat = np.array(bn)
    h, pic_len = bn_mat.shape
    project = np.sum(bn_mat, 1)
    pos = np.where(project <= split_threshold)[0]
    if len(pos) == 0 or pos[0] != 0:
        pos = np.insert(pos, 0, 0)
    if pos[-1] != len(project):
        pos = np.append(pos, len(project))
    diff = np.diff(pos)

    if len(diff) == 0:
        return [[np.array(image), (0, 0, pic_len, h)]]

    width = np.max(diff)
    coordinate = list(zip(pos[:-1], pos[1:]))
    info = list(zip(diff, coordinate))
    info = list(filter(lambda x: x[0] > np.min(diff)*3, info))

    split_pos = []
    temp = []
    for pos_info in info:
        if width-2 <= pos_info[0] <= width:
            if temp:
                split_pos.append(temp.pop(0))
            split_pos.append(pos_info)

        elif pos_info[0] < width-2:
            temp.append(pos_info)
            if len(temp) > 1:
                s, e = temp[0][1][0], temp[1][1][1]
                if e - s <= width + 2:
                    temp = [(e - s, (s, e))]
                else:
                    split_pos.append(temp.pop(0))

    if temp:
        split_pos.append(temp[0])

    # crop images with split_pos
    line_res = []
    if blank:
        if len(split_pos) == 1:
            pos_info = split_pos[0][1]
            ymin, ymax = max(0, pos_info[0]-2), min(h, pos_info[1]+2)
            return [[np.array(image.crop((0, ymin, pic_len, ymax))), (0, ymin, pic_len, ymax)]]

        length = len(split_pos)
        for i in range(length):
            if i == 0:
                next_info = split_pos[i+1]
                margin = min(next_info[1][0] - split_pos[i][1][1], 2)
                ymin, ymax = max(0, split_pos[i][1][0] - margin), split_pos[i][1][1] + margin
                x1, y1, x2, y2 = 0, ymin, pic_len, ymax
                sub = image.crop((x1, y1, x2, y2))
            elif i == length-1:
                pre_info = split_pos[i - 1]
                margin = min(split_pos[i][1][0] - pre_info[1][1], 2)
                ymin, ymax = split_pos[i][1][0] - margin, min(h, split_pos[i][1][1] + margin)
                x1, y1, x2, y2 = 0, ymin, pic_len, ymax
                sub = image.crop((x1, y1, x2, y2))
            else:
                next_info = split_pos[i + 1]
                pre_info = split_pos[i - 1]
                margin = min(split_pos[i][1][0] - pre_info[1][1], next_info[1][0] - split_pos[i][1][0], 2)
                ymin, ymax = split_pos[i][1][0] - margin, split_pos[i][1][1] + margin
                x1, y1, x2, y2 = 0, ymin, pic_len, ymax
                sub = image.crop((x1, y1, x2, y2))

            line_res.append([np.array(sub), (x1, y1, x2, y2)])
    else:
        for pos_info in split_pos:
            x1, y1, x2, y2 = 0, pos_info[1][0], pic_len, pos_info[1][1]
            sub = image.crop((x1, y1, x2, y2))
            line_res.append([np.array(sub), (x1, y1, x2, y2)])

    return line_res


def _groupify(myarray, torlerence, groupify_method=np.mean, inplace=False):
    groups = DBSCAN(eps=torlerence, min_samples=1).fit_predict(myarray.reshape(-1, 1))
    array_copy = myarray if inplace else myarray.copy()
    for l in np.unique(groups):
        array_copy[groups == l] = groupify_method(array_copy[groups == l])
    return array_copy


def line_split_fine_print(image, n_font_sizes = 1, table=TABLE, split_threshold=0, margin_ratio=0.1, **_):
    if not isinstance(image, Image.Image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            raise TypeError

    image_ = image.convert('L')
    bn = image_.point(table, '1')
    bn_mat = np.array(bn)
    h, pic_len = bn_mat.shape
    project = np.sum(bn_mat, 1)
    mask = (project > split_threshold).astype(int)
    diff = mask - shift(mask, 1, cval=0)
    content_t = np.where(diff==1)[0]
    content_b = np.where(diff==-1)[0]
    line_heights = content_b - content_t
    group_tol = np.ceil(min(line_heights)*0.5)
    horizontal_ranges = np.array(
        [np.where(bn_mat[t:b + 1].sum(axis=0) > 0)[0][[0, -1]] for t, b in zip(content_t, content_b)])
    content_l, content_r = horizontal_ranges.T
    line_heights_grouped = _groupify(line_heights, group_tol, np.max)
    content_l_grouped = _groupify(content_l, group_tol, np.min)
    content_r_grouped = _groupify(content_r, group_tol, np.max)

    standard_height = np.partition(line_heights_grouped, -n_font_sizes)[0]
    line_heights_grouped = np.maximum(standard_height, line_heights_grouped)

    margin = int(np.floor(standard_height*margin_ratio))
    l_min = min(content_l_grouped)
    r_max = max(content_r_grouped)

    def adjust_box(l, t, r, b, target_height, l_limit, t_limit, r_limit, b_limit, margin, l_min=0, r_max=0, tol=0):
        l_final = max(l_limit, l-margin)
        r_final = min(r_limit, int(r+margin+target_height*0.7))
        if b - t < target_height:
            if (l > l_min + tol and r < r_max - tol) or (l < l_min + tol and r > r_max - tol):
                mid = (b_limit + t_limit) + 2
                if b + margin >= np.floor(mid + target_height / 2):
                    b = min(b_limit - 1, b + margin)
                    return l_final, b - target_height, r_final, b
                elif t - margin <= np.ceil(mid - target_height / 2):
                    t = max(t_limit + 1, t - margin)
                    return l_final, t, r_final, t + target_height
                else:
                    b = np.floor(mid + target_height / 2)
                    return l_final, b - target_height, r_final
            if l > l_min + tol and b-t < 0.25*target_height:
                raise RuntimeError(f"整行标点符号？？？ {l} {t} {r} {b}")
            if t - t_limit < b_limit - (t + target_height):
                t = max(t_limit + 1, t - margin)
                return l_final, t, r_final, t + target_height
            b = min(b_limit - 1, b + margin)
            return l_final, b - target_height, r_final, b
        return l_final, max(t_limit, t-margin), r_final, min(b_limit, b+margin)

    box = adjust_box(
        content_l_grouped[0], content_t[0], content_r_grouped[0], content_b[0], line_heights_grouped[0],
        0, 0, pic_len, content_t[1] if len(content_t)>1 else h, margin, l_min=l_min, r_max=r_max, tol=0.5*standard_height
    )
    res = [(np.array(image.crop(box)), box)]
    if len(content_l_grouped) == 1:
        return first_line
    for i in range(1, len(content_l_grouped)-1):
        box = adjust_box(
            content_l_grouped[i], content_t[i], content_r_grouped[i], content_b[i], line_heights_grouped[i],
            0, content_b[i-1], pic_len, content_t[i+1], margin, l_min=l_min, r_max=r_max, tol=0.5 * standard_height
        )
        res.append((np.array(image.crop(box)), box))
    box = adjust_box(
        content_l_grouped[-1], content_t[-1], content_r_grouped[-1], content_b[-1], line_heights_grouped[-1],
        0, content_b[-2], pic_len, h, margin, l_min=l_min, r_max=r_max, tol=0.5 * standard_height
    )
    res.append((np.array(image.crop(box)), box))
    return res
