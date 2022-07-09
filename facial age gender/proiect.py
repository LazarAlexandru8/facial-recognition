
#importarea bibliotecilor
import cv2
import pafy

#preluarea unui link si creerea unui obiect "video" pentru rularea acestuia cu OpenCV

link = 'https://www.youtube.com/watch?v=ZxQevO0nwUw'
paf = pafy.new(link)
video = paf.getbest(preftype="mp4")

box = cv2.VideoCapture(video.url)

#crearea a 3 liste pentru depozitare

mean = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 3)', '(4, 7)', '(8, 14)', '(15, 24)', '(25, 37)', '(38, 47)', '(48, 59)', '(60, 100)']
gender_list = ['Barbat', 'Femeie']

box.set(3, 640)  # setarea latimii ferestrei
box.set(4, 860)  # setarea inaltimii ferestrei



#definirea unei functii "modele" care incarca caffemodel si prototxt pentru detectarea varstei si genului

def modele():

    age_m = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')

    gender_m = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')

    return (age_m, gender_m)

def citire_fata(age_m, gender_m):
    font = cv2.FONT_HERSHEY_COMPLEX

    while True:

        #citirea imaginii preluata cu VideoCapture

        ret, image = box.read()

        #incarcarea modelelor pentru recunoastere faciala

        detectare_faciala = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

        #convertirea imaginii intr-una grayscale deoarece OpenCV necesita o imagine grayscale pentru recunoasterea faciala

        gri = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #crearea unui obiect pentru detectarea fetei si returnarea unei liste de pozitii de tipul (x,y,w,h)

        faces = detectare_faciala.detectMultiScale(gri, 1.1, 5)


        for (x, y, w, h) in faces:

            #desenarea unei rame in jurul fetei

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

            # preluarea fetei
            face_img = image[y:y + h, h:h + w].copy()

            #preprocesarea fetei

            prep = cv2.dnn.blobFromImage(face_img, 1, (227, 227), mean, swapRB=False)

            # indentificarea genului
            gender_m.setInput(prep)
            gender_x = gender_m.forward()
            gender = gender_list[gender_x[0].argmax()]

            # indentificarea varstei
            age_m.setInput(prep)
            age_x = age_m.forward()
            age = age_list[age_x[0].argmax()]


            #crarea unui text pentru gen si varsta in casuta videoului

            textvid = "%s %s" % (gender, age)
            cv2.putText(image, textvid, (x, y), font, 1, (255, 225, 255), 2, cv2.LINE_AA)


        #afisarea rezultatului final

        cv2.imshow('frame', image)


        #oprirea programului la detectarea tastei "x"

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break


#programul main

if __name__ == "__main__":
    age_m, gender_m = modele()

    citire_fata(age_m, gender_m)
