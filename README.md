# Hill Climb Rac — Mini Game Berbasis Warna Tangan

Game hill climb sederhana yang dimainkan dengan tangan di depan kamera. Tangan kanan diarahkan ke kotak GAS, tangan kiri ke kotak REM — sama seperti konsep mini game PCV lain, tapi kali ini satu tangan satu pedal.

Game bisa juga dicoba penuh pakai keyboard tanpa kamera.

## Cara kerja

Kamera menangkap gambar, lalu OpenCV mengubahnya ke HSV dan mencari warna biru. Objek biru terbesar yang terdeteksi dianggap tangan (maksimal dua). Area kanan bawah frame kamera adalah pedal gas, area kiri bawah adalah pedal rem — tangan yang masuk ke area tersebut menekan pedalnya, ditandai lingkaran hijau di layar game.

Fisika mobil disimulasikan di Pygame: gravitasi, tanjakan, lompatan natural saat melewati puncak, bahan bakar yang terus berkurang saat digas, koin untuk skor, serta tombol mundur untuk keluar dari jurang. Ada dua peta: Countryside (mudah) dan Cliff (curam). Rekor skor dan jarak terbaik tiap peta otomatis tersimpan ke `savegame.json`, jadi tetap ada walaupun aplikasinya ditutup.

## Menjalankan

Pasang dependensi:

```
pip install pygame-ce opencv-python numpy
```

Jalankan game:

```
python main/main.py
```

Untuk mencoba tanpa kamera, jalankan lewat keyboard:

```
python main/main.py --no-hsv
```

Kontrol:

| Aksi | Kamera | Keyboard |
|---|---|---|
| Gas | tangan kanan ke kotak GAS | panah kanan/atas atau D |
| Rem | tangan kiri ke kotak REM | panah kiri/bawah atau A |
| Mundur | — | S |
| Pilih menu | tahan tangan di tombol | mouse / panah + Enter |
| Jeda | — | Esc |

Menu dipilih dengan menahan tangan di atas tombol sampai progressnya penuh (atau pakai mouse kalau tanpa kamera).

## Aset

Gambar PNG disimpan di folder `assets/`:

- `car_body.png`, `car_wheel.png`, `driver.png` untuk mobil dan pengemudi
- `Coin/image_1.png` – `image_4.png` untuk animasi koin
- `gas/image_1.png` – `image_4.png` untuk animasi bahan bakar
- `home_bg.png`, `map_cliff_bg.png`, `map_countryside_bg.png` untuk latar
- `ground_cliff.png`, `ground_countryside.png` untuk tekstur tanah

Kalau ada gambar yang tidak tersedia, game otomatis memakai tampilan sederhana sebagai pengganti.
