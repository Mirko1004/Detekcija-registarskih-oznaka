
import cv2
import os
import numpy as np

import DetectChars
import DetectPlates
import PossiblePlate

# varijable modula
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False


def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # pokušaj KNN-a

    if blnKNNTrainingSuccessful == False:                               # ako KNN trening nije bio uspjesan
        print("\nerror: KNN traning was not successful\n")  # pokazi gresku
        return                                                          # i zatvori program

    imgOriginalScene  = cv2.imread("LicPlateImages/auto1.jpeg")               # otvaramo sliku

    if imgOriginalScene is None:                            # ako slika nije ucitana uspjesno
        print("\nerror: image not read from file \n\n")  # pokazi gresku
        os.system("pause")                                  # pauziraj da korisnik mote vidjeti gresku
        return                                              # i zatvori program

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detektiraj registraciju

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detektiraj karakter u registraciji

    cv2.imshow("imgOriginalScene", imgOriginalScene)            # pokazi scene sliku

    if len(listOfPossiblePlates) == 0:                          # ako nije nadena nijedna registracija
        print("\nno license plates were detected\n")  #  informiraj korisnika da nije nadena
    else:
                # ako lista mogucih registracijskih oznaka ima bar jednu registraciju

                #  sortiraj listu mogucih reg.oznaka silazno (najveci broj karakteka prema najmanjem broju karaktera)
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)


    # pretpostavimo da je tablica sa najvise prepoznatih karaktera (znaci tablica koja je u sortiranju prva) nasa stvarna tablica
        licPlate = listOfPossiblePlates[0]

        cv2.imshow("imgPlate", licPlate.imgPlate)           # pokazi odrezanu tablicu i pragove tablice
        cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:                     # ako nisu pronadeni karakteri u tablici
            print("\nno characters were detected\n\n")  # pokazi poruku da nisu pronadeni
            return                                          # zatvori program

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # crtamo crveni pravokutnik oko tablice

        print("\nlicense plate read from image = " + licPlate.strChars + "\n")  # napisi tekst registracijske tablice
        print("----------------------------------------")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # napisi tekst registracijske tablice na sliku

        cv2.imshow("imgOriginalScene", imgOriginalScene)                # ponovo pokazi scenu slike

        cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           # napisi sliku u datoteku



    cv2.waitKey(0)					# drzi prozor otvoren sve dok korisnik ne prisitne tipku

    return



def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # dohvati 4 vrha rotiranog pravokutnika

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # nacrtaj 4 crvene linije
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function


def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             # ovo ce biti centar podrucja na kojem ce tekst biti napisan
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          # ovo ce biti donji lijevi kut podrucja gdje ce se pisati tekst
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # izaberemo font
    fltFontScale = float(plateHeight) / 30.0                    # osnovna skala fonta na visini tablice
    intFontThickness = int(round(fltFontScale * 1.5))           # debljina fonta

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        # pozovi getTextSize

            # raspakiraj rotirani pravokutnik u centar, visina i sirina i kut
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              # moramo biti sigurni da je centar integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         # horizontalna lokacija podrucja teksta je ista kao i tablica

    if intPlateCenterY < (sceneHeight * 0.75):                                                  # ako se registracijska tablica nalazi u gornje 3/4 slike
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # napisi karaktere ispod tablice
    else:                                                                                       # inace ako je registracijska tablica u donjem dijelu 1/4 slike
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # napisi karaktere iznad tablice

    textSizeWidth, textSizeHeight = textSize                # raspakiraj velicinu i sirinu teksta

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           # izracunaj donji lijevi kut podrucja teksta
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          # baziran na centar, visinu i sirinu podrucja teksta

            # napisi tekst na sliku
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)
# end function


if __name__ == "__main__":
    main()
