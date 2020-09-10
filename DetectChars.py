
import os

import cv2
import numpy as np
import math
import random

import Main
import Preprocess
import PossibleChar

# varijable modula

kNearest = cv2.ml.KNearest_create()

        # konstante za checkIfPossibleChar, ovo provjerava samo jedan moguci karakter(ne usporeduje ga s drugim karakterom)
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

        # konstante za usporedivanje 2 karaktera
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

        # druge konstante
MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100


def loadKNNDataAndTrainKNN():
    allContoursWithData = []                # deklariramo praznu listu
    validContoursWithData = []              # uskoro cemo je napuniti

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # ucitaj klasifikacije
    except:                                                                                 # ako datoteka ne moze biti otvorena
        print("error, unable to open classifications.txt, exiting program\n")  # pokazi gresku s porukom
        os.system("pause")
        return False                                                                        # i vrati false
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # ucitaj slike
    except:                                                                                 # ako datoteka ne moze biti otvorena
        print("error, unable to open flattened_images.txt, exiting program\n")  # pokazi gresku s porukom
        os.system("pause")
        return False                                                                        # i vrati false
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # preoblikuj numpy array na 1, potrebno za proci poziv do trraina

    kNearest.setDefaultK(1)                                                             # postavi zadani K na 1

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)           # train KNN

    return True                             # ako smo dosli do ovdje training je bio uspjesan, tako da vracamo True
# end function


def detectCharsInPlates(listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossiblePlates) == 0:          # ako je lista mogucih tablica prazna
        return listOfPossiblePlates             # return
    # end if

            # ako smo dosli do ovog dijela, mozemo biti sigurni da lista mogucih tablica sadrzi najmanje jednu tablicu

    for possiblePlate in listOfPossiblePlates:          # za svaku mogucu tablicu, ovo je velika for petlja koja sadrzi vecinu funkcija

        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Preprocess.preprocess(possiblePlate.imgPlate)     # predproces za dobiti sivi ton i prag slika

        if Main.showSteps == True: # pokazi korake
            cv2.imshow("5a", possiblePlate.imgPlate)
            cv2.imshow("5b", possiblePlate.imgGrayscale)
            cv2.imshow("5c", possiblePlate.imgThresh)


                # povecaj velicinu slike tablia da bi lakse vidjeli i detektirali karaktere
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)

                # opet radimo pragove slika da eliminiramo siva podrucja
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if Main.showSteps == True: # pokazi korake
            cv2.imshow("5d", possiblePlate.imgThresh)


                # nadi sve moguce karaktere na tablici
                # ova funkcija prvo pronalazi sve konture, nakon toga ukljucuje samo one konture koje mogu biti karakteri(bez usporedivanja sa drugim karakterima,zasad)
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        if Main.showSteps == True: # pokazi korake
            height, width, numChannels = possiblePlate.imgPlate.shape
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]                                         # ocisti listu kontura

            for possibleChar in listOfPossibleCharsInPlate:
                contours.append(possibleChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)

            cv2.imshow("6", imgContours)
        # pokazi korake

                # data je lista svih mogucih karaktera, nadi grupu odgovarajucih znakova unutar tablica
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

        if Main.showSteps == True: # pokazi korake
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for
                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("7", imgContours)
        # end if # pokazi korake

        if (len(listOfListsOfMatchingCharsInPlate) == 0):			# ako u tablici nije nadena grupa odgovarajucih znakova

            if Main.showSteps == True: # pokazi korake
                print("chars found in plate number " + str(
                    intPlateCounter) + " = (none), click on any image and press a key to continue . . .")
                intPlateCounter = intPlateCounter + 1
                cv2.destroyWindow("8")
                cv2.destroyWindow("9")
                cv2.destroyWindow("10")
                cv2.waitKey(0)
            # end if # pokazi korake

            possiblePlate.strChars = ""
            continue						# idi nazad na vrh for petlje
        # end if

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):                              # unutar svake liste odgovarajucih znakova
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)        # sortiraj znakove slijeva na desno
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])              # i ukloni unutarnje preklapajuce znakove
        # end for

        if Main.showSteps == True: # pokazi korake
            imgContours = np.zeros((height, width, 3), np.uint8)

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                del contours[:]

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for

                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("8", imgContours)
        # end if # pokazi korake

                # unutar svake moguce tablice, pretpostavimo da je najduza lista odgovarajucih mogucih znakova,nasa aktualna lista
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

                # petlja kroz sve vektore odgovarajucih znakova, uzmi indeks jedne sa najvise znakova
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
            # end if
        # end for

                # pretpostavimo da je najduza lista odgovarajucih mogucih znakova unutar tablice zapravo nasa aktualna lista znakova
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        if Main.showSteps == True: # pokazi korake
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for matchingChar in longestListOfMatchingCharsInPlate:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)

            cv2.imshow("9", imgContours)
        # end if # pokazi korake

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)

        if Main.showSteps == True: # pokazi korake
            print("chars found in plate number " + str(
                intPlateCounter) + " = " + possiblePlate.strChars + ", click on any image and press a key to continue . . .")
            intPlateCounter = intPlateCounter + 1
            cv2.waitKey(0)
        # end if # pokazi korake

    # zavrsetak velike for petlje u kojoj se nalazi vecina funkcija

    if Main.showSteps == True:
        print("\nchar detection complete, click on any image and press a key to continue . . .\n")
        cv2.waitKey(0)
    # end if

    return listOfPossiblePlates
# end function


def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []                        # ovo ce biti return vrijednost
    contours = []
    imgThreshCopy = imgThresh.copy()

            # nadi sve konture u tablici
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:                        # za svaku konturu
        possibleChar = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(possibleChar):              # ako je kontura moguci znak, zabiljezi to i ne usporeduj sa ostalim znakovima (zasad)
            listOfPossibleChars.append(possibleChar)       # dodaj u listu mogucih znakova
        # end if
    # end if

    return listOfPossibleChars
# end function


def checkIfPossibleChar(possibleChar):
            # ova funkcija je prva koja radi grubu provjeru da vidi da li kontura moze biti znak
            # zapamtimo da jos nismo usporedili znak sa drugim znakovima da trazimo u grupama
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False
    # end if
# end function


def findListOfListsOfMatchingChars(listOfPossibleChars):
            # sa ovom funkcijom, pocinjemo sa svim mogucim znakovima u jednoj velikoj listi
            # svrha ove funkcije je preurediti veliku listu znakova u listu odgovarajucih znakova
            # moramo znati da znakovi koji nisu nadeni u grupi odgovarajucih ne moraju biti razmatrani dalje
    listOfListsOfMatchingChars = []                  # ovo ce biti return vrijednost

    for possibleChar in listOfPossibleChars:                        # za svaki moguci znak u velikoj listi znakova
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)        # nadi sve moguce znakove u velikoj listi koje odgovaraju trenutnom znaku

        listOfMatchingChars.append(possibleChar)                # takoder dodaj trenutni znak u trenutnu mogucu listu odgovarajucih znakova

        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     # ako trenutna moguca lista odgovarajucih znakova nije dovoljno duga da bi konstituirali mogucu tablicu
            continue                            # skoci nazad na vrh for petlje i pokusav ponovno sa sljedecim znakom, ali to nije potrebno
                                                # za spremiti listu u bilo kojem nacinu dok nema dovoljno znakova za mogucu tablicu
        # end if

                                                # ako smo dosli do ovdje, treuntna lista je prosla test kao grupa ili grupiranje odgovarajucih znakova
        listOfListsOfMatchingChars.append(listOfMatchingChars)      # zato dodamo u nasu listu od liste odgovarajucih znakova

        listOfPossibleCharsWithCurrentMatchesRemoved = []

                                                # uklonimo trenutnu listu odgovarajucih znakova iz velike liste tako da ne koristimo iste znakove dva puta
                                                # budimo sigurni da napravimo novu veliku listu za ovo jer ne zelimo promijeniti originalnu veliku listu
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      # rekurzija

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        # za svaku listu odgovarajucih znakova nadenu pomocu rekurzije
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             # dodaj u nasu originalnu listu od liste odgovarajucih znakova
        # end for

        break       # exit for

    # end for

    return listOfListsOfMatchingChars
# end function


def findListOfMatchingChars(possibleChar, listOfChars):
            # svrha ove funkcije je, dati moguci znak i veliku listu mogucih znakova
            # nadi sve znakove u velikoj listi koji odgovaraju jednom mogucem znaku, i vrati te odgovarajuce znakove kao listu
    listOfMatchingChars = []                # ovo ce biti return vrijednost

    for possibleMatchingChar in listOfChars:                # za svaki znak u velikoj listi
        if possibleMatchingChar == possibleChar:    # ako je znak za koji pokusavamo naci odgovarajuci znak isti kao i znak u velikoj listi koji trenutno i provjeravamo
                                                    # onda ga ne smijemo ukljuciti u listu odgovarajucih znakova zato sto ce to zavrsiti kao duplanje trenutnog znaka
            continue                                # zato ga ne dodajemo u listu odgovarajucih znakova i vracamo se nazad na vrh for petlje
        # end if
                    # izracunamo stvari da vidimo jesu li znakovi odgovarajuci
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

                # gledamo da li znakovi odgovaraju
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            listOfMatchingChars.append(possibleMatchingChar)        # ako znakovi odgovaraju, dodaj trenutni znak u listu odgovarajucih znakova
        # end if
    # end for

    return listOfMatchingChars                  # vrati rezultat
# end function


# koristimo Pitagorin teorem za izracunati udaljenost izmedu dva znaka
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))
# end function


# koristimo osnovnu trigonometriju za izracunati kut izmedu znakova
def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:                           # moramo provjeriti da slucajno ne dijelimo s nulom ako je centar x pozicije jednak, jer dijeljenje s 0 u Pythonu ce izbaciti gresku
        fltAngleInRad = math.atan(fltOpp / fltAdj)      # ako nije nula, izracunaj kut
    else:
        fltAngleInRad = 1.5708                          # ako je 0, koristi ovo kao kut,
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       # izracunaj rezultat u stupnjevima

    return fltAngleInDeg
# end function


# ako imamo dva znaka koji se preklapaju ili su preblizu jedan drugome da eventualno budu odvojeni znakovi, ukloni manji znak
# ovo je zbog prevencije kako se neki znak ne bi pojavio duplo ako su dvije konture nadene za isti znak
# npr za slovo O oba unutarnja i vanjska prstena mogu biti nadeni kao konture, ali mi moramo koristiti znak samo jednom
def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)                # ovo ce biti return vrijednost

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:        # ako trenutni znak i drugi znak nisu isti
                                                                            # ako trenutni znak i drugi znak imaju centar skoro na istoj lokaciji
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                                # ako smo dosli ovdje, nasli smo znakove koji se preklapaju
                                # zatim moramo identificirati koji znak je manji, zatim ako taj znak nije vec uklonjen, uklonimo ga
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:         # ako je trenutni znak manji od drugog znaka
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:              # ako trenutni znak nije vec uklonjen u prethodnoj provjeri
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)         # ukloni trenutni znak
                        # end if
                    else:                                                                       # inace ako je drugi znak manji od trenutnog znaka
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:                # ako drugi znak vec nije uklonjen u prethodnoj provjeri
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)           # ukloni drugi znak
                        # end if
                    # end if
                # end if
            # end if
        # end for
    # end for

    return listOfMatchingCharsWithInnerCharRemoved
# end function


# ovdje primjenjujemo trenutno raspoznavanje znaka
def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""               # ovo ce biti return vrijednost, znakovi u registracijskoj tablici

    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sortiraj znakove slijeva na desno

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)                     # napravi verziju u boji od praga slike tako da mozemo nacrtati konture u boji

    for currentChar in listOfMatchingChars:                                         # za svaki znak u tablici
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2, Main.SCALAR_GREEN, 2)           # nacrtaj zelenu kutiju oko znaka

                # izrezi znak izvan praga slike
        imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))           # promijeni velicinu slike, ovo je neophodno za detekciju znakova

        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))        # izravnaj sliku u 1 numpy array

        npaROIResized = np.float32(npaROIResized)               # konvertiraj sa 1 numpy array integera, u 1 numpy array floats

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)              # konacno mozemo pozvati findNearest

        strCurrentChar = str(chr(int(npaResults[0][0])))            # uzmemo znak iz rezultata

        strChars = strChars + strCurrentChar                        # dodamo trenutni znak u puni string

    # end for

    if Main.showSteps == True: # koraci
        cv2.imshow("10", imgThreshColor)
    # end if # pokazi korake

    return strChars
# end function