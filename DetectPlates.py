
import cv2
import numpy as np
import math
import Main
import random

import Preprocess
import DetectChars
import PossiblePlate
import PossibleChar

# varijable modula
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5


def detectPlatesInScene(imgOriginalScene):
    listOfPossiblePlates = []                   # ovo ce biti return vrijednost

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    if Main.showSteps == True: # pokazi korake
        cv2.imshow("0", imgOriginalScene)
    # end if # pokazi korake

    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(imgOriginalScene)         # predproces za dobiti sivi ton i prag slika

    if Main.showSteps == True: # pokazi korake
        cv2.imshow("1a", imgGrayscaleScene)
        cv2.imshow("1b", imgThreshScene)
    # end if # pokazi korake

    # nadi sve moguce karaktere na tablici
    # ova funkcija prvo pronalazi sve konture, nakon toga ukljucuje samo one konture koje mogu biti karakteri(bez usporedivanja sa drugim karakterima,zasad)
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)

    if Main.showSteps == True: # pokazi korake
        print("step 2 - len(listOfPossibleCharsInScene) = " + str(
            len(listOfPossibleCharsInScene)))

        imgContours = np.zeros((height, width, 3), np.uint8)

        contours = []

        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)
        # end for

        cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
        cv2.imshow("2b", imgContours)
    # end if # pokazi korake

            # data je lista svih mogucih karaktera, nadi grupu odgovarajucih znakova unutar tablica
            # u sljedecim koracima svaka grupa odgovarajucih znakova ce  se pokusati prepoznati kao tablica
    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    if Main.showSteps == True: # pokazi korake
        print("step 3 - listOfListsOfMatchingCharsInScene.Count = " + str(
            len(listOfListsOfMatchingCharsInScene)))

        imgContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
        # end for

        cv2.imshow("3", imgContours)
    # end if # pokazi korake

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:                   # za svaku grupu odgovarajucih znakova
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)         # pokusaj izvuci tablicu

        if possiblePlate.imgPlate is not None:                          # ako je tablica nadena
            listOfPossiblePlates.append(possiblePlate)                  # dodaj na listu mogucih tablica
        # end if
    # end for

    print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")

    if Main.showSteps == True: # pokazi korake
        print("\n")
        cv2.imshow("4a", imgContours)

        for i in range(0, len(listOfPossiblePlates)):
            p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)

            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), Main.SCALAR_RED, 2)

            cv2.imshow("4a", imgContours)

            print("possible plate " + str(i) + ", click on any image and press a key to continue . . .")

            cv2.imshow("4b", listOfPossiblePlates[i].imgPlate)
            cv2.waitKey(0)
        # end for

        print("\nplate detection complete, click on any image and press a key to begin char recognition . . .\n")
        cv2.waitKey(0)
    # end if # pokazi korake

    return listOfPossiblePlates
# end function


def findPossibleCharsInScene(imgThresh):
    listOfPossibleChars = []                # ovo ce biti return vrijednost

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()

    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # nadi sve konture

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):                       # za svaku konturu

        if Main.showSteps == True: # pokazi korake
            cv2.drawContours(imgContours, contours, i, Main.SCALAR_WHITE)
        # end if # pokazi korake

        possibleChar = PossibleChar.PossibleChar(contours[i])

        if DetectChars.checkIfPossibleChar(possibleChar):                   # ako je kontura moguci znak, zabiljezi to i ne usporeduj sa ostalim znakovima (zasad)
            intCountOfPossibleChars = intCountOfPossibleChars + 1           # povecaj broj mogucih znakova
            listOfPossibleChars.append(possibleChar)                        # dodaj u listu mogucih znakova
        # end if
    # end for

    if Main.showSteps == True: # pokazi korake
        print("\nstep 2 - len(contours) = " + str(len(contours)))
        print("step 2 - intCountOfPossibleChars = " + str(intCountOfPossibleChars))
        cv2.imshow("2a", imgContours)
    # end if # pokazi korake

    return listOfPossibleChars
# end function


def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()           # ovo ce biti return vrijednost

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sortiraj znakove slijeva nadesno bazirano na x poziciju

            # izracunaj poziciju centra tablice
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

            # izracunaj sirinu i visinu tablice
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    # end for

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

            # izracunaj ispravljeni kut tablice
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

            # zapakiraj centar tablice, sirinu i visinu, i ispravjeni kut u rotiranu varijablu clana pravokutnika
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

            # zadnji koraci su za izvodenje rotacije

            # uzmimo rotaciju za nasu izracunatu korekciju kuta
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape      # raspakiraj originalnu sirinu i visinu slike

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))       # zarotiraj cijelu sliku

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possiblePlate.imgPlate = imgCropped         # kopiraj isjecenu sliku tablice unutar primjenjive varijable clana moguce tablice

    return possiblePlate
# end function