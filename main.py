import numpy as np
import cv2 as cv

print('openCV version: ',cv.__version__)


def reshapeImg(img, size_w):
    img_w = img.shape[1]
    img_h = img.shape[0]
    img_ratio = img_h / img_w

    new_h = int(img_ratio * size_w)

    dst = cv.resize(img, (size_w, new_h))
    print(dst.shape)

    return dst


def setRect(img, pts):
    (x, y, w, h) = cv.boundingRect(pts)

    (x, y, w, h) = (int(x),
                    int(y),
                    int(w),
                    int(h))

    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cv.rectangle(img, pt1, pt2, (0, 255, 0), 4)


def pTransform(img, src_pts, pts, resizeScale = 1):
    h = int(cv.minAreaRect(pts)[1][0] * resizeScale)
    w = int(cv.minAreaRect(pts)[1][1] * resizeScale)

    p_ltop = [0, 0]
    p_rtop = [w - 1, 0]
    p_llow = [0, h - 1]
    p_rlow = [w - 1, h - 1]

    src_pts = np.array(src_pts).astype(np.float32)
    #원본 이미지에서 추적된 4점입니다.
    dst_pts = np.array([p_ltop, p_rtop, p_llow, p_rlow]).astype(np.float32)
    #minAreaRect의 모양대로 perspective warping을 하기 위한 목표가 되는 직사각형의 네 모서리 좌표들입니다.

    pers_mat = cv.getPerspectiveTransform(src_pts, dst_pts)

    dst = cv.warpPerspective(img, pers_mat, (w, h))

    return dst


# https://github.com/ashay36/Document-Scanner/blob/master/document_scanner.py

def step1(src, dst): # 자동으로 사각형의 4 모서리를 찾아 장방형으로 출력해주는 함수입니다.
    origin = src.copy()

    # 외곽선 검출을 위한 전처리
    # 1. 사이즈 조정
    resizeW = 500 # 외곽선 검출을 더 용이하게 하기 위해 resize를 진행하는데, 그 기준이 되는 width의 크기를 지정하는 변수입니다.
    src = reshapeImg(src, resizeW) #src의 width값이 resizeW와 같고 원본 영상의 비율을 유지해 확대, 축소합니다.

    # 2. 그레이스케일 변환, noise 제거를 위한 Gaussian Blur 필터 사용, Canny 엣지 검출
    src_gray = cv.cvtColor(src.copy(), cv.COLOR_BGR2GRAY) #에지 검출을 위해 그레이스케일로 변환해줍니다.
    src_gray_blurred = cv.GaussianBlur(src_gray, (5, 5), 0) #가우시안 블러를 사용해 노이즈를 제거해줍니다.
    src_gray_blurred_canny = cv.Canny(src_gray_blurred, 80, 150)# Canny 에지 검출

    # 3. Contour 검출
    contours, _ = cv.findContours(src_gray_blurred_canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #contour 검출
    #print('area', contours)
    #contours = sorted(contours, key=cv.contourArea, reverse=True)[:5] #contour들 중 contourArea가 가장 큰
    #print(cv.contourArea(contours))
   # contours = np.array(contours)
   # print('dddd', contours.shape)

    for pts in contours:
        if cv.contourArea(pts) < 1000: #영역이 일정 이하면 노이즈로 간주하고 continue해줍니다.
            continue

        # Douglas Peucker 알고리즘을 이용해 외곽선을 근사화 한다.
        approx = cv.approxPolyDP(pts, cv.arcLength(pts, True) * 0.02, True)

        approx = np.reshape(approx, (4, 2))#array를 편히 다루기 위해 shpae을 정리한다.

        #이후 perspective warping을 사용하기 위해 좌표를 정리한다.
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


            resizeToOriginVal = origin.shape[1]/resizeW
            print('ddd', resizeToOriginVal)
            approx = np.multiply(approx, resizeToOriginVal)
            approx = np.array(approx, dtype=int)
            #원본의 전처리를 위해 사이즈를 줄였고 그 영상에서 4점의 좌표를 얻었습니다.
            #원본 사진을 perspective warping하기 위해 그 점들을 원본좌표의 스케일로 좌표들을 변환해줍니다.

            #print('app1', origin.shape[1] / resizeW)

            sccanedObject = pTransform(origin, approx, pts, resizeScale= 1)#DP알고리즘에 의해 근사화된 4점과 피사체를 둘러싸고 있는 minAreaRec()의 4점을 perspective warp해줍니다.
            setRect(dst, approx)  # Test용 직사각형
            for i in range(4):
                cv.circle(origin, (approx[i][0], approx[i][1]), 10, (255, 0, 0), 3)

            print('find rectangular')

            cv.imshow('sccanedObject',sccanedObject)
            break


#originalImg = cv.imread('card.png', cv.IMREAD_COLOR) #원본소스로 항상 가져다 쓸 수 있게 이후에 이미지를 복사해서 쓰도록 한다.
originalImg = cv.imread('receipt22.png', cv.IMREAD_COLOR)



if originalImg is None:
    print('Loading Error')
    exit(0)

src = originalImg.copy()

step1(src, src)

cv.imshow('src', src)

cv.waitKey(0)
cv.destroyAllWindows()
