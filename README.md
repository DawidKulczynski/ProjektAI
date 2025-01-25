# Projekt AI - Age Detection App
Aplikacja webowa do przewidywania wieku na podstawie zdjęcia twarzy, wykorzystująca Flask, TensorFlow i OpenCV.

# Wymagania i uruchomienie
Program korzysta z bibliotek które nie wspierają wersji Python nowszych niż 3.11

Wszystkie biblioteki znajdują się w pliku requirements.txt i można je zainstalować przy pomocy ```pip install -r requirements.txt```

Dataset należy pobrać z https://drive.google.com/drive/folders/1HROmgviy4jUUUaCdvvrQ8PcqtNg2jn3G  i party rozpakować do folderu UTKFace w miejscu projektu w taki sposób:

![image](https://github.com/user-attachments/assets/9e38f844-9efa-4955-ad10-93d12f5230e8)

Plik ```index.html``` należy dodatkowo przenieść do folderu templates w miejscu projektu

![image](https://github.com/user-attachments/assets/d6510810-7004-4229-8c13-bb8ee6c45c26)

Po spełnieniu tych warunków program jest gotowy do uruchomienia.

Po uruchomieniu można otworzyć przeglądarkę i przejść do http://127.0.0.1:5000/

