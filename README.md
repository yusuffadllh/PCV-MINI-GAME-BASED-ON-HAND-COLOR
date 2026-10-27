# Hill Climb Rac 

Game hill climb sederhana yang dimainkan dengan tangan di depan kamera. Tangan kanan diarahkan ke kotak GAS, tangan kiri ke kotak REM
Game bisa juga dicoba penuh pakai keyboard tanpa kamera.

## Cara kerja

Kamera menangkap gambar, lalu OpenCV mengubahnya ke HSV dan mencari warna biru. Objek biru terbesar yang terdeteksi dianggap tangan (maksimal dua). Area kanan bawah frame kamera adalah pedal gas, area kiri bawah adalah pedal rem, tangan yang masuk ke area tersebut menekan pedalnya, ditandai lingkaran hijau di layar game.

## Menjalankan
Pasang dependensi:

```
pip install pygame-ce opencv-python numpy
```

Jalankan game:

```
python main/main.py

```

## Aset

Gambar PNG disimpan di folder `assets/`:

- `car_body.png`, `car_wheel.png`, `driver.png` untuk mobil dan pengemudi
- `Coin/image_1.png` – `image_4.png` untuk animasi koin
- `gas/image_1.png` – `image_4.png` untuk animasi bahan bakar
- `home_bg.png`, `map_cliff_bg.png`, `map_countryside_bg.png` untuk latar
- `ground_cliff.png`, `ground_countryside.png` untuk tekstur tanah

