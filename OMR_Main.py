import cv2
import numpy as np
import utlis


########################################################################
webCamFeed = True
pathImage = "tl7.jpg"


heightImg = 700
widthImg  = 700
questions=5
choices=5

########################################################################


count=0



img = cv2.imread(pathImage)

img = cv2.resize(img, (widthImg, heightImg)) # THAY ĐỔI KÍCH THƯỚC HÌNH ẢNH
imgFinal = img.copy()
imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) #TẠO HÌNH ẢNH NGÂN HÀNG ĐỂ KIỂM TRA GỬI GỠ NẾU BẮT BUỘC
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CHUYỂN ĐỔI HÌNH ẢNH THÀNH QUY MÔ XÁM
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # THÊM GAUSSIAN BLUR
imgCanny = cv2.Canny(imgBlur,10,70) # APPLY CANNY

cv2.imshow("gus",imgGray)
cv2.imshow("gus1",imgBlur)

imgContours = img.copy() # SAO CHÉP HÌNH ẢNH ĐỂ HIỂN THỊ MỤC ĐÍCH
imgBigContour = img.copy() # CSAO CHÉP HÌNH ẢNH ĐỂ HIỂN THỊ MỤC ĐÍCH
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # lọc đối tượng
cv2.drawContours(imgContours, contours, -1, (0, 0, 255), 20) # VẼ TẤT CẢ CÁC KẾT NỐI ĐÃ PHÁT HIỆN
print("contours[0]")

print(contours)
cv2.imshow("contours",imgContours)
rectCon = utlis.rectContour(contours) # BỘ LỌC CHO CÁC CONTOURS RECTANGLE
biggestPoints= utlis.getCornerPoints(rectCon[0]) # NHẬN ĐIỂM  HÌNH LỚN NHẤT
print("biggestPoints")

print(biggestPoints)
print("biggestPoints")


if biggestPoints.size != 0:

            # diện tích hình chữ nhật  lớn nhất
    biggestPoints=utlis.reorder(biggestPoints) # sắp xếp lại theo giá trị
    cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20) # VẼ CONTOUR LỚN NHẤT
    pts1 = np.float32(biggestPoints) # điểm bắt đầu 1
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # điểm bắt đầu 2
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # NHẬN ma trận  CHUYỂN ĐỔI
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) # ạp dụng vào ảnh ban đầu


# áp dụng phân ngưỡng
imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY) # CHUYỂN ĐỔI THÀNH XÁM
imgThresh = cv2.threshold(imgWarpGray, 170, 255,cv2.THRESH_BINARY_INV )[1] # áp dụng  phân ngưỡng và hàm ngược
cv2.imshow("a ", imgThresh)
boxes = utlis.splitBoxes(imgThresh) # lấy từng ô trong ảnh
cv2.imshow("Split Test ", boxes[0])
countR=0
countC=0
myPixelVal = np.zeros((questions,choices)) # ĐỂ LƯU TRỮ CÁC GIÁ TRỊ KHÔNG CÓ GIÁ TRỊ CỦA MỖI HỘP
print(myPixelVal)
for image in boxes:

    totalPixels = cv2.countNonZero(image)
    cv2.imshow("b ", totalPixels)
    myPixelVal[countR][countC]= totalPixels
    countC += 1
    if (countC==choices):countC=0;countR +=1

            # TÌM CÂU TRẢ LỜI CỦA NGƯỜI DÙNG VÀ ĐĂNG HỌ VÀO DANH SÁCH
myIndex=[]
for x in range (0,questions):
    arr = myPixelVal[x]
    myIndexVal = np.where(arr == np.amax(arr))
    myIndex.append(myIndexVal[0][0])


utlis.showAnswers(imgWarpColored,myIndex) # VẼ CÂU TRẢ LỜI ĐÃ PHÁT HIỆN

cv2.imshow("Final Result", imgFinal)
cv2.imshow("test", imgWarpColored)
cv2.waitKey(0)



