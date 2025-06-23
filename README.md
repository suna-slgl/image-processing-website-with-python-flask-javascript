# 🖼️ Image Processing Website with Python Flask & JavaScript

Bu proje, Python'un Flask framework'ü ve JavaScript kullanılarak geliştirilmiş bir görüntü işleme web sitesidir. Kullanıcılar, görsellerini yükleyip çeşitli görüntü işleme işlemlerini web arayüzü üzerinden kolayca gerçekleştirebilirler.

## 🎨 Özellikler

- Görüntü yükleme ve önizleme
- Görüntüyü gri tonlamaya çevirme
- Görüntüyü döndürme (sağa/sola)
- Görüntüyü yeniden boyutlandırma
- Görüntüyü kırpma
- Görüntüyü keskinleştirme veya bulanıklaştırma
- Renk filtreleri uygulama
- Görüntüyü yansıtma (ayna, üstten-alta, soldan-sağa)
- Üzerine yazı ekleme
- İşlenen görseli indirme seçeneği
- Flask ile RESTful API
- JavaScript ile dinamik ve hızlı kullanıcı arayüzü

## 🛠️ Kurulum

1. **Depoyu klonlayın:**
   ```bash
   git clone https://github.com/suna-slgl/image-processing-website-with-python-flask-javascript.git
   cd image-processing-website-with-python-flask-javascript
   ```

2. **Sanal ortam oluşturun ve başlatın:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows için: venv\Scripts\activate
   ```

3. **Gerekli Python paketlerini yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Uygulamayı başlatın:**
   ```bash
   python app.py
   ```
   veya
   ```bash
   flask run
   ```

5. **Web sitesine erişin:**
   - Tarayıcınızda `http://localhost:5000` adresine gidin.

## 👩‍💻 Kullanım

1. Görselinizi yükleyin.
2. İstediğiniz işlemi seçin (ör. gri tonlama, döndürme, filtre uygulama, kırpma, vb.).
3. Sonucu önizleyin ve gerekirse indirin.

## 🗂️ Proje Yapısı

```
image-processing-website-with-python-flask-javascript/
│
├── app.py
├── script.js
├── style.css
├── index.html
├── requirements.txt
└── README.md
```

## 📦 Gereksinimler

- Python 3.7+
- Flask
- Pillow ve/veya OpenCV (görüntü işleme için)
- JavaScript (istemci tarafı işlemler için)

Gerekli tüm Python paketleri `requirements.txt` dosyasında listelenmiştir.
