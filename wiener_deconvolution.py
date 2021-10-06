
import numpy as np
import cv2


def blur_edge(img, d=31):  # размытие краев
    h, w = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2 * d + 1, 2 * d + 1), -1)[d:-d, d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w - x - 1, y, h - y - 1]).min(-1)
    w = np.minimum(np.float32(dist) / d, 1.0)
    return img * w + img_blur * (1 - w)


def motion_kernel(angle, d, sz=65): # перерисовка линейной psf, учитывая градус поворота и размер
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:, 2] = (sz2, sz2) - np.dot(A[:, :2], ((d - 1) * 0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern


def defocus_kernel(d, sz=65): # перерисока psf в виде окружности, учитывая новый диаметр
    kern = np.zeros((sz, sz), np.uint8)
    cv2.circle(kern, (sz, sz), d, 255, -1, cv2.LINE_AA, shift=1)
    kern = np.float32(kern) / 255.0
    return kern


if __name__ == '__main__':
    print(__doc__)
    import sys

    try:
        fn = sys.argv[1]
    except:
        print("Нет изображения! Программа завершена! ")
        exit(-1)

    win = 'deconvolution'

    img = cv2.imread(fn, 0)
    if img is None:
        print('Failed to load fn:', fn)
        sys.exit(1)

    img = np.float32(img) / 255.0
    cv2.imshow('input', img)

    img = blur_edge(img)
    IMG = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

    defocus = True


    def update(_):
        ang = np.deg2rad(cv2.getTrackbarPos('angle', win))
        d = cv2.getTrackbarPos('d', win)
        noise = 10 ** (-0.1 * cv2.getTrackbarPos('SNR (db)', win))

        if defocus:
            psf = defocus_kernel(d)
        else:
            psf = motion_kernel(ang, d)
        cv2.imshow('psf', psf)

        psf /= psf.sum()
        psf_pad = np.zeros_like(img)
        kh, kw = psf.shape
        psf_pad[:kh, :kw] = psf
        PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows=kh)  # дискретное преобразование Фурье
        PSF2 = (PSF ** 2).sum(-1)
        iPSF = PSF / (PSF2 + noise)[..., np.newaxis]
        RES = cv2.mulSpectrums(IMG, iPSF, 0)
        # поэлементное умножение двух комплексных матриц, которые являются
        # результатом действительного или комплексного преобразования Фурье. Ну или свертка.
        res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        # Вычисляет обратное дискретное преобразование Фурье для одномерного или двухмерного массива.
        res = np.roll(res, -kh // 2, 0)
        res = np.roll(res, -kw // 2, 1)
        cv2.imshow(win, res)


    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('psf', 0)
    cv2.createTrackbar('angle', win, 135, 180, update)
    cv2.createTrackbar('d', win, 22, 50, update)
    cv2.createTrackbar('SNR (db)', win, 25, 50, update)
    update(None)

    while True:
        ch = cv2.waitKey()
        if ch == 27:
            break
        if ch == ord(' '):
            defocus = not defocus
            update(None)
