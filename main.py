import copy
import random

from PIL import Image
from skimage.io import imread, imsave, imshow, show
from skimage import util, filters
import scipy.fftpack as fp
from scipy import ndimage, misc, signal
from skimage import data, img_as_float
from skimage.color import rgb2gray
from skimage.transform import rescale
import matplotlib.pylab as pylab
import numpy as np
import numpy.fft
import timeit
import PIL.ImageStat as stat

#一维信号平滑
# pointNumber: int = 500
# t = np.linspace(0, 2*np.pi, 500)
# y = np.sin(t)
# noise = 0.4*np.random.random(500)
# y = y + noise
# y1 = copy.deepcopy(y)
# fig = pylab.figure(figsize=(12, 10))
# pylab.subplot(2, 2, 1)
# pylab.plot(t, y)
# pylab.xlabel('t')
# pylab.title('SIN(t) NOISE ADDED')
#
# wndsize: int = 9
# for i in range(wndsize//2, pointNumber - wndsize//2):
#     _ = 0
#     for j in range(-wndsize//2, wndsize//2):
#         _ = _ + y[i+j]
#     y1[i] = _ / wndsize
#
# pylab.subplot(2, 2, 2)
# pylab.plot(t, y1)
# pylab.xlabel('t')
# pylab.title('SHIFT SMOOTH')
#
# myFilter = np.ones(wndsize) / wndsize
# yy = signal.convolve(y, myFilter, 'same')
# pylab.subplot(2, 2, 3)
# pylab.plot(t, yy)
# pylab.xlabel('t')
# pylab.title('CONVOVLE')
#
# pylab.subplot(2, 2, 4)
# pylab.plot(t, y1, 'g', label='SMOOTH')
# pylab.plot(t, yy, 'r', label='CONV')
# pylab.legend(loc=0)
# pylab.xlabel('t')
# pylab.title('COMPARE')
# fig.show()
# fig.savefig('/Users/zhuyu/Desktop/signalProject/image/singleDimension.png')


#二维图像
# im1 = rgb2gray(imread('/Users/zhuyu/Desktop/signalProject/image/rabbit.jpeg'))
# fig = pylab.figure(figsize=(5, 8))

#二维傅里叶变换和反变换
# freq1 = fp.fft2(im1)
# im1_ = fp.ifft2(freq1).real
#
# pylab.subplot(2, 2, 1)
# pylab.imshow(im1, cmap='gray')
# pylab.title('Original Image', size=20)
#
# pylab.subplot(2, 2, 2)
# pylab.imshow(20 * np.log10(0.01 + np.abs(fp.fftshift(freq1))), cmap='gray')
# pylab.title('FFT Spectrum Magnitude', size=20)
#
# pylab.subplot(2, 2, 3)
# pylab.imshow(np.angle(fp.fftshift(freq1)), cmap='gray')
# pylab.title('FFT Phase', size=20)
#
# pylab.subplot(2, 2, 4)
# pylab.imshow(np.clip(im1_, 0, 255), cmap='gray')
# pylab.title('Reconstructed Image', size=20)
# pylab.show()
#
# fig.savefig('/Users/zhuyu/Desktop/signalProject/image/rabbitFig.png')

#卷积操作的平滑和锐化处理
# scharr = np.array([[-0.8, 0 - 2j, +0.8],
#                    [-2 + 0j, 0 + 0j, +2 + 0j],
#                    [-0.8, 0 + 2j, +0.8]])  # Gx + j*Gy
# smooth = np.array([[0.1, 0.1, 0.1],
#                    [0.1, 0.1, 0.1],
#                    [0.1, 0.1, 0.1]])
# grad = signal.convolve2d(im1, scharr, boundary='symm', mode='same')
# smoothed = signal.convolve2d(im1, smooth, boundary='symm', mode='same')
# fig, (ax_orig, ax_mag1, ax_mag2) = pylab.subplots(3, 1, figsize=(6, 15))
# ax_orig.imshow(im1, cmap='gray')
# ax_orig.set_title('Original')
# ax_orig.set_axis_off()
# ax_mag1.imshow(np.absolute(grad), cmap='gray')
# ax_mag1.set_title('Gradient magnitude - SHARP')
# ax_mag1.set_axis_off()
# ax_mag2.imshow(np.absolute(smoothed), cmap='gray')
# ax_mag2.set_title('Gradient magnitude - SMOOTH')
# ax_mag2.set_axis_off()
# ax_ang.imshow(np.angle(grad), cmap='hsv') # hsv is cyclic, like angles
# ax_ang.set_title('Gradient orientation')
# ax_ang.set_axis_off()


# fig.show()
# fig.savefig('/Users/zhuyu/Desktop/signalProject/image/rabbitConv2d_new.png')


#Sa函数卷积
# [row, col] = im1.shape
# span = 20
# ax = np.linspace(-span//2, span//2, span)
# photo = np.zeros((span, span))
# fsinc = np.sinc(np.pi*ax)
# for i in range(0, span):
#     for j in range(0, span):
#         photo[i][j] = (1 / span**2 * fsinc[i]*fsinc[j])
#
# im = signal.convolve2d(im1, photo, boundary='symm', mode='same')
# pylab.subplot(2, 1, 1)
# pylab.imshow(im, cmap='gray')
# pylab.subplot(2, 1, 2)
# pylab.imshow(20*np.log(0.01 + np.abs(fp.fftshift(fp.fft2(im)))), cmap='gray')
# fig.show()
# fig.savefig('/Users/zhuyu/Desktop/signalProject/image/convBySinc.png')

# #直接构造一个频谱的滤波器smooth
# fig = pylab.figure(figsize=(12, 10))
# [row, col] = im1.shape
# smooth = np.zeros((row, col))
#
# for i in range(row//2-20, row//2+20):
#     for j in range(col//2-20, col//2+20):
#         smooth[i][j] = 1
#
# freq1 = fp.fft2(im1)
# freq1_shifted = fp.fftshift(freq1)
#
# freq = np.multiply(freq1_shifted, smooth)
#
# pylab.subplot(2, 1, 1)
# pylab.imshow(20 * np.log10(0.01 + np.abs(freq)), cmap='gray')
# pylab.title('FFT Spectrum Magnitude', size=20)
#
# pylab.subplot(2, 1, 2)
# im1_ = fp.ifft2(freq).real
# pylab.imshow(np.clip(im1_, 0, 255), cmap='gray')
# pylab.title('FFT Recovery', size=20)
#
# fig.savefig('/Users/zhuyu/Desktop/signalProject/image/multipleByFreq.png')


#将时域kernel转化到频域再相乘
# fig = pylab.figure(figsize=(12, 10))
# [row, col] = im1.shape
# smooth = np.zeros((row, col))
#
# for i in range(row//2-5, row//2+5):
#     for j in range(col//2-5, col//2+5):
#         smooth[i][j] = 0.5
#
# freq1 = fp.fft2(im1)
# freq2 = fp.fft2(smooth)
# freq = np.multiply(freq1, freq2)
#
# pylab.subplot(2, 1, 1)
# pylab.imshow(20 * np.log10(0.01 + np.abs(fp.fftshift(freq))), cmap='gray')
# pylab.title('FFT Spectrum Magnitude', size=20)
#
# pylab.subplot(2, 1, 2)
# im1_ = fp.ifft2(freq).real
# pylab.imshow(np.clip(fp.fftshift(im1_), 0, 255), cmap='gray')
# pylab.title('FFT Recovery', size=20)
#
# fig.show()
# fig.savefig('/Users/zhuyu/Desktop/signalProject/image/ChangeIntoFreqFirst.png')


#两者比较
#
# fig = pylab.figure(figsize=(12, 10))
# [row, col] = im1.shape
# smooth1 = np.zeros((row, col))
# span = 5
# for i in range((row - span)//2, (row + span)//2):
#     for j in range((col - span)//2, (col + span)//2):
#         smooth1[i][j] = 1 / span**2
#
# freq1 = fp.fft2(im1)
# freq2 = fp.fft2(smooth1)
# freq = np.multiply(freq1, freq2)
#
# pylab.subplot(2, 2, 1)
# im1_ = fp.ifft2(freq).real
# pylab.imshow(np.clip(fp.fftshift(im1_), 0, 255), cmap='gray')
# pylab.title('FFT Recovery', size=15)
#
# pylab.subplot(2, 2, 2)
# pylab.imshow(20 * np.log10(0.01 + np.abs(fp.fftshift(freq))), cmap='gray')
# pylab.title('FFT Spectrum Magnitude', size=15)
#
# smooth2 = np.ones((span, span)) / span**2
# smoothed = signal.convolve2d(im1, smooth2, mode='full')
# pylab.subplot(2, 2, 3)
# pylab.imshow(smoothed, cmap='gray')
# pylab.title('CONV Recovery', size=15)
#
# freq = fp.fftshift(fp.fft2(smoothed))
# pylab.subplot(2, 2, 4)
# pylab.imshow(20 * np.log10(0.01 + np.abs(freq)), cmap='gray')
# pylab.title('FFT Spectrum Magnitude', size=15)
#
# fig.show()
# fig.savefig('/Users/zhuyu/Desktop/signalProject/image/convTheory_2.png')

#研究kernel性质
# fig2 = pylab.figure(figsize=(12, 10))
# pylab.subplot(2, 2, 1)
# pylab.imshow(20 * np.log10(0.01 + np.abs(fp.fftshift(fp.fft2(smooth2)))), cmap='gray')
# pylab.subplot(2, 2, 2)
# pylab.imshow(20 * np.log10(0.01 + np.abs(fp.fftshift(freq2))), cmap='gray')
# pylab.subplot(2, 2, 3)
# pylab.imshow(smooth2, cmap='gray')
# pylab.subplot(2, 2, 4)
# pylab.imshow(smooth1, cmap='gray')
# fig2.show()

#kernel扩大
# fig3 = pylab.figure(figsize=(10, 12))
# filter = np.zeros((row, col))
# for m in range(1, 5):
#     span = 10 * m
#     for i in range(row//2 - span//2, row//2 + span//2):
#         for j in range(col//2 - span//2, col//2 + span//2):
#             filter[i][j] = 1 / span**2
#
#     f = fp.fft2(filter)
#     pylab.subplot(4, 3, 3*m-1)
#     pylab.imshow(20 * np.log10(0.01 + np.abs(fp.fftshift(f))), cmap='gray')
#     pylab.subplot(4, 3, 3*m-2)
#     pylab.imshow(filter, cmap='gray')
#     pylab.subplot(4, 3, 3*m)
#     pylab.plot(np.abs(fp.fftshift(f[row//2])))
#
# fig3.show()
# fig3.savefig('/Users/zhuyu/Desktop/signalProject/image/spanningKernelFFT.png')

#sinc函数傅里叶变换
fig = pylab.figure(figsize=(8, 5))
size = 20
# span = 10
ax = np.linspace(-2, 2, size)
photo = np.zeros((size, size))
fsinc = np.sinc(2*ax)
for i in range(0, size):
    for j in range(0, size):
        photo[i][j] = fsinc[i]*fsinc[j]

pylab.subplot(1, 2, 1)
pylab.imshow(photo, cmap='gray')
pylab.title('Sinc(2x)')
pylab.subplot(1, 2, 2)
pylab.imshow(20*np.log(0.01+np.abs(fp.fftshift(fp.ifft2(photo)))), cmap='gray')
pylab.title('IFFT')
fig.show()
fig.savefig('/Users/zhuyu/Desktop/信号分析与处理/大作业-平滑、滤波和降噪/image/sinc2D.png')


# #扩大卷积核
# fig = pylab.figure(figsize=(12, 10))
# [row, col] = im1.shape
# fig = pylab.figure(figsize=(12, 10))
#
# smooth = np.ones((5, 5))
# smoothed1 = signal.convolve2d(im1, smooth, boundary='symm', mode='same')
# pylab.subplot(2, 2, 1)
# pylab.imshow(smoothed1, cmap="gray")
# pylab.title('5X5 kernel')
#
# smooth2 = np.ones((20, 20))
# smoothed2 = signal.convolve2d(im1, smooth2, boundary='symm', mode='same')
# pylab.subplot(2, 2, 2)
# pylab.imshow(smoothed2, cmap="gray")
# pylab.title('20X20 kernel')
#
# smooth3 = np.ones((40, 40))
# smoothed3 = signal.convolve2d(im1, smooth3, boundary='symm', mode='same')
# pylab.subplot(2, 2, 3)
# pylab.imshow(smoothed3, cmap="gray")
# pylab.title('40X40 kernel')
#
# smooth4 = np.ones((80, 80))
# smoothed4 = signal.convolve2d(im1, smooth4, boundary='symm', mode='same')
# pylab.subplot(2, 2, 4)
# pylab.imshow(smoothed3, cmap="gray")
# pylab.title('80X80 kernel')
#
# fig.show()
# fig.savefig('/Users/zhuyu/Desktop/signalProject/image/kernelGrowLarger.png')

#加椒盐校验噪声
# fig = pylab.figure(figsize=(10, 15))
# [row, col] = im1.shape
# blank = np.zeros((row, col))
# # prob = 0.05
# # for i in range(0, row):
# #     for j in range(0, col):
# #         rdm = random.random()
# #         if rdm < prob:
# #             noise[i][j] = 0
# #         elif rdm > (1 - prob):
# #             noise[i][j] = 255
# #         else:
# #             noise[i][j] = im1[i][j]
# noise = util.random_noise(blank, mode='s&p')
# pylab.subplot(3, 2, 1)
# pylab.imshow(noise, cmap="gray")
# pylab.title("NOISE")
#
# pylab.subplot(3, 2, 2)
# pylab.imshow(im1, cmap="gray")
# pylab.title("ORIGIN")
#
# # im1 = im1 + noise
# im2 = util.random_noise(im1, mode='s&p')
# pylab.subplot(3, 2, 3)
# pylab.imshow(im2, cmap="gray")
# pylab.title("ADD NOISE")
#
# smooth = np.ones((5, 5))
# smoothed = signal.convolve2d(im2, smooth, boundary='symm', mode='same')
# pylab.subplot(3, 2, 4)
# pylab.imshow(smoothed, cmap="gray")
# pylab.title("FILTERED by conv 5X5")
#
# smooth = np.ones((10, 10))
# smoothed = signal.convolve2d(im2, smooth, boundary='symm', mode='same')
# pylab.subplot(3, 2, 5)
# pylab.imshow(smoothed, cmap="gray")
# pylab.title("FILTERED by conv 10X10")
#
# im3 = filters.median(im2, np.ones((3, 3)))
# pylab.subplot(3, 2, 6)
# pylab.imshow(im3, cmap="gray")
# pylab.title("FILTERED by median")
#
# fig.show()
# fig.savefid('/Users/zhuyu/Desktop/signalProject/image/addNoiseAndFiltered.png')

# fig2 = pylab.figure(figsize=(12, 10))
# freq = fp.fft2(noise)
# pylab.imshow(20 * np.log10(0.01 + np.abs(fp.fftshift(freq))), cmap='gray')
# fig2.show()