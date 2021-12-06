import pytesseract as pt
from pytesseract import Output
import numpy as np
import cv2 as cv

def reshapeImg(img, size_w):  # width를 지정하면 그 크기를 기준으로 비율에 맞춰 크기 변환, return = reshapeImg(변환할 이미지, width)
    img_w = img.shape[1]
    img_h = img.shape[0]
    img_ratio = img_h / img_w

    new_h = int(img_ratio * size_w)

    dst = cv.resize(img, (size_w, new_h), interpolation=cv.INTER_AREA)

    return dst


def test_step1(img, pts):  # 테스트를 위한 boundRect와 네 꼭지점을 화면에 표시해줍니다.
    tmp = img.copy()

    setRect(tmp, pts)
    for i in range(4):
        cv.circle(tmp, (pts[i][0], pts[i][1]), 10, (0, 0, 255), 15)
    cv.imshow('test_step1', tmp)


def setRect(img, pts):  # setRect(ContourRect를 그릴 원본 이미지, 네 점들), 테스트용 함수입니다.
    (x, y, w, h) = cv.boundingRect(pts)

    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cv.rectangle(img, pt1, pt2, (0, 255, 0), 4)


def pTransform(img, src_pts, resizeScale=1):
    # return = pTransform(변환할 이미지, minAreaRect를 추적할 원본 오브젝트의 4점, 변환 후 스케일 조정을 위한 인자)
    h = int(cv.minAreaRect(src_pts)[1][0] * resizeScale)
    w = int(cv.minAreaRect(src_pts)[1][1] * resizeScale)

    p_ltop = [0, 0]
    p_rtop = [w - 1, 0]
    p_llow = [0, h - 1]
    p_rlow = [w - 1, h - 1]

    src_pts = np.array(src_pts).astype(np.float32)
    # 원본 이미지에서 추적된 4점입니다.
    dst_pts = np.array([p_ltop, p_rtop, p_llow, p_rlow]).astype(np.float32)
    # minAreaRect의 모양대로 perspective warping을 하기 위한 목표가 되는 직사각형의 네 모서리 좌표들입니다.

    pers_mat = cv.getPerspectiveTransform(src_pts, dst_pts)
    # perspective Transform을 위한 변환행렬을 구합니다.

    dst = cv.warpPerspective(img, pers_mat, (w, h))
    # perspective 변환
    return dst


def step1(src):  # 자동으로 사각형의 4 모서리를 찾아 장방형으로 출력해주는 함수입니다. step1(스캔할 원본 이미지)
    origin = src.copy()

    # 외곽선 검출을 위한 전처리
    # 1. 사이즈 조정
    resizeW = 500  # 외곽선 검출을 더 용이하게 하기 위해 resize를 진행하는데, 그 기준이 되는 width의 크기를 지정하는 변수입니다.
    src = reshapeImg(src.copy(), resizeW)  # src의 width값이 resizeW와 같고 원본 영상의 비율을 유지해 확대, 축소합니다.

    # 2. 그레이스케일 변환, noise 제거를 위한 Gaussian Blur 필터 사용, Canny 엣지 검출
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)  # 에지 검출을 위해 그레이스케일로 변환해줍니다.
    src_gray_blurred = cv.GaussianBlur(src_gray, (5, 5), 0)  # 가우시안 블러를 사용해 노이즈를 제거해줍니다.
    src_gray_blurred_canny = cv.Canny(src_gray_blurred, 80, 150)  # Canny 에지 검출

    # 3. Contour 검출
    contours, _ = cv.findContours(src_gray_blurred_canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # contour 검출
    # print('area', contours)
    # contours = sorted(contours, key=cv.contourArea, reverse=True)[:5] #contour들 중 contourArea가 가장 큰
    # print(cv.contourArea(contours))
    # contours = np.array(contours)
    # print('dddd', contours.shape)

    for pts in contours:
        if cv.contourArea(pts) < 1000:  # 영역이 일정 이하면 노이즈로 간주하고 continue해줍니다.
            continue

        # Douglas Peucker 알고리즘을 이용해 외곽선을 근사화 한다.
        approx = cv.approxPolyDP(pts, cv.arcLength(pts, True) * 0.02, True)

        approx = np.reshape(approx, (4, 2))  # array를 편히 다루기 위해 shpae을 정리한다.

        # 이후 perspective warping을 사용하기 위해 좌표를 정리한다.
        approx = np.array(sorted(approx, key=lambda x: x[1]))  # 상부 하부 정렬을 위해 y축 좌표값을 기준으로 정렬
        approx[:2] = np.array(sorted(approx[:2], key=lambda x: x[0]))  # 상부 x정렬 기준으로 정렬
        approx[2:4] = np.array(sorted(approx[2:4], key=lambda x: x[0]))  # 하부 x정렬 기준으로 정렬
        # sort 후 approx
        # [[좌, 상단]
        # [[우, 상단]
        # [[좌, 하단]
        # [[우, 하단]]

        vtc = len(approx)
        if vtc == 4:
            # print(approx)

            resizeToOriginVal = origin.shape[1] / resizeW
            approx = np.multiply(approx, resizeToOriginVal)
            approx = np.array(approx, dtype=int)
            # 원본의 전처리를 위해 사이즈를 줄였고 그 영상에서 4점의 좌표를 얻었습니다.
            # 원본 사진을 perspective warping하기 위해 그 점들을 원본좌표의 스케일로 좌표들을 변환해줍니다.

            sccanedObject = pTransform(origin, approx, resizeScale=resizeToOriginVal)
            # DP알고리즘에 의해 근사화된 4점과 피사체를 둘러싸고 있는 minAreaRec()의 4점을 perspective warp해줍니다.

            # test_step1(origin, approx) #test용 직사각형과 네 점을 표시하는 함수입니다.

            print('\nstep1: found rectangle - 피사체 장방형 변환 완료')

            return sccanedObject

def step2(src): # OCR 글자인식을 위한 전처리 함수
    # 전처리 순서: 사이즈 조정, 그레이스케일 변환, 노이즈 제거를 위한 GaussianBlur(), 영상 이진화, 이진화 이후 생긴 Salt Pepper 잡음 제거를 위한 medianBlur()
    origin = src.copy()

    # OCR 글자 인식을 위한 전처리
    # 1. 사이즈 조정
    resizeW = 800  # 연산 속도 개선과 글자 인식을을 더 높게하게 하기 위해 resize를 진행하는데, 그 기준이 되는 width의 크기를 지정하는 변수입니다.
    origin = reshapeImg(origin, resizeW)  # src의 width값이 resizeW와 같고 원본 영상의 비율을 유지해 확대, 축소합니다.

    # 2. 그레이스케일 변환, noise 제거를 위한 Gaussian Blur 필터 사용, 영상 이진화, 모폴로지 연산 수행
    src_gray = cv.cvtColor(origin, cv.COLOR_BGR2GRAY)  # 그레이스케일로 변환해줍니다.
    src_gray_blurred = cv.GaussianBlur(src_gray, (3, 3), 1)  # 가우시안 블러를 사용해 노이즈를 제거해줍니다.
    binarizedImg = cv.adaptiveThreshold(src_gray_blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 13, 9) # 적응형 이진화를 사용해줍니다.


    binarizedImg_blurred = cv.medianBlur(binarizedImg, 5) #Salt Pepper 잡음 제거를 위한 블러

    kernel = np.ones((3, 3), np.uint8)
    src_morp = cv.morphologyEx(binarizedImg_blurred, cv.MORPH_OPEN, kernel, iterations=1)

    cv.imshow('Pretreated Image', src_morp)

    return src_morp, origin


def step3(src, filename):
    pretreatedImg, origin = step2(src)

    ocr_text = pt.image_to_string(pretreatedImg, lang='kor+eng')  # OCR으로 문자를 인식합니다.

    print('\n----------- OCR 처리 결과입니다. -----------')  # OCR 결과를 콘솔에 출력해줍니다.
    print(ocr_text)


    d = pt.image_to_data(pretreatedImg, output_type=Output.DICT) # bounding box 표시를 위한 data를 return하는 함수입니다.
    # Bounding box를 생성하는 함수는 외부자료를 참고한 부분입니다.
    # 출처: https://stackoverflow.com/questions/20831612/getting-the-bounding-box-of-the-recognized-words-using-python-tesseract
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv.rectangle(origin, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #cv.imshow('Bounding Box image', origin)

    filename_f, filename_r = filename.split('.')
    outputFilename = filename_f + '_output.' + filename_r # 파일이름에 _output을 붙여주는 부분입니다.
    cv.imwrite(outputFilename,origin) # 바운딩 박스 처리 된 이미지를 파일이름_output.확장자 로 파일을 출력해줍니다.





def main():

    filename = 'issac.png'
    originalImg = cv.imread(filename, cv.IMREAD_COLOR)

    if originalImg is None:
        print('Loading Error')
        exit(0)
    # 파일을 입력하고 에러 판단을 합니다.


    src = originalImg.copy()
    cv.imshow('Original Image', src)  # 원본 이미지입니다.

    # step 1

    sccanedObject = step1(src) # 입력한 원본 이미지에서 사각 피사체의 네 모서리를 인식해 장방형으로 변환해줍니다.
    cv.imshow('Scanned Object', sccanedObject)  # 자동으로 네점을 추적해 장방형으로 처리된 이미지를 화면에 표시합니다.

    # step 2, 3

    step3(sccanedObject, filename)# 함수 내에서 글자 인식을 위한 전처리 단계에 해당하는 step2()를 수행 후
    # OCR로 글씨 인식 및 Bounding box로 표시한 이미지를 출력하는 함수입니다.



    step2(sccanedObject)

    cv.waitKey(0)
    cv.destroyAllWindows()


main()
