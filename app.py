import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/adaptive-thresholding', methods=['POST'])
def adaptive_thresholding_route():
    data = request.json.get('image')
    if not data:
        return jsonify({'message': 'Resim yüklenmedi!'}), 400
    
    # Base64 verisini numpy array'e dönüştür
    image_data = base64.b64decode(data.split(",")[1])
    image = Image.open(BytesIO(image_data))
    image = np.array(image)

    # Renkli resimse gri tonlamaya çevir
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image

    # Adaptive Thresholding işlemi
    adaptive_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    # İşlenmiş resmi Base64 formatına geri çevir
    _, buffer = cv2.imencode('.png', adaptive_image)
    buffer = BytesIO(buffer)
    processed_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return jsonify({"message": "Adaptive Thresholding completed", "image": processed_image_base64})

# Bulanıklaştırma işlevi
@app.route('/blur', methods=['POST'])
def blur_image():
    data = request.json.get('image')
    if not data:
        return jsonify({"message": "Resim yüklenmedi!"}), 400
    
    # Base64 verisini numpy array'e dönüştür
    image_data = base64.b64decode(data.split(",")[1])  # Veriyi base64'ten çözüp byte dizisine dönüştür
    image = Image.open(BytesIO(image_data))  # Resmi aç
    image = np.array(image)  # Resmi numpy dizisine dönüştür

    # Gri tonlamaya çevir (renkli ise)
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image

    # Gaussian Blur işlemi
    blurred_image = cv2.GaussianBlur(gray_image, (15, 15), 0)  # Bulanıklaştırma işlemi

    # İşlenmiş resmi Base64 formatına geri çevir
    _, buffer = cv2.imencode('.png', blurred_image)  # PNG formatında kodlama
    buffer = BytesIO(buffer)
    processed_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')  # Base64'e çevir

    return jsonify({"message": "Blurring completed", "image": processed_image_base64})

@app.route('/sharpness', methods=['POST'])
def sharpness_image():
    data = request.json.get('image')
    if not data:
        return jsonify({"message": "Resim yüklenmedi!"}), 400
    
    try:
        # Base64 verisini numpy array'e dönüştür
        image_data = base64.b64decode(data.split(",")[1])
        image = Image.open(BytesIO(image_data))
        image = np.array(image)

        # Gri tonlamaya çevir
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image

        # Keskinleştirme işlemi: Unsharp Mask yöntemi
        kernel = np.array([[-1, -1, -1], 
                           [-1,  9, -1], 
                           [-1, -1, -1]])  # Bu kernel keskinleştirme işlemi yapar
        sharp_image = cv2.filter2D(gray_image, -1, kernel)

        # İşlenmiş resmi Base64 formatına geri çevir
        _, buffer = cv2.imencode('.png', sharp_image)
        buffer = BytesIO(buffer)
        processed_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({"message": "Sharpness completed", "image": processed_image_base64})

    except Exception as e:
        return jsonify({"message": f"Bir hata oluştu: {str(e)}"}), 500

@app.route('/gamma-filter', methods=['POST'])
def gamma_filter():
    data = request.json.get('image')
    if not data:
        return jsonify({"message": "Resim yüklenmedi!"}), 400

    try:
        # Base64 verisini numpy array'e dönüştür
        image_data = base64.b64decode(data.split(",")[1])
        image = Image.open(BytesIO(image_data))
        image = np.array(image)

        # Gri tonlamaya çevir
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image

        # Sabit bir gamma değeri
        gamma = 1.5  # Burada sabit bir gamma değeri kullanılıyor
        gamma_corrected_image = np.array(255 * (gray_image / 255) ** gamma, dtype='uint8')

        # İşlenmiş resmi Base64 formatına geri çevir
        _, buffer = cv2.imencode('.png', gamma_corrected_image)
        buffer = BytesIO(buffer)
        processed_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({"message": "Gamma Filtering completed", "image": processed_image_base64})

    except Exception as e:
        print(f"Error during Gamma Filtering: {str(e)}")  # Detaylı hata mesajını yazdır
        return jsonify({"message": f"Bir hata oluştu: {str(e)}"}), 500

    data = request.json.get('image')
    gamma_value = request.json.get('gamma', 1.0)  # Default gamma 1.0
    if not data:
        return jsonify({"message": "Resim yüklenmedi!"}), 400
    
    try:
        # Base64 verisini numpy array'e dönüştür
        image_data = base64.b64decode(data.split(",")[1])
        image = Image.open(BytesIO(image_data))
        image = np.array(image)

        # Renkli resimse gri tonlamaya çevir
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Gamma Filterleme işlemi
        gamma_correction_image = np.power(image / 255.0, gamma_value) * 255.0
        gamma_correction_image = np.uint8(gamma_correction_image)

        # İşlenmiş resmi Base64 formatına geri çevir
        _, buffer = cv2.imencode('.png', gamma_correction_image)
        buffer = BytesIO(buffer)
        processed_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({"message": "Gamma Correction completed", "image": processed_image_base64})

    except Exception as e:
        return jsonify({"message": f"Bir hata oluştu: {str(e)}"}), 500

@app.route('/canny', methods=['POST'])
def canny_filter():
    data = request.json.get('image')
    if not data:
        return jsonify({"message": "Resim yüklenmedi!"}), 400

    try:
        # Base64 verisini numpy array'e dönüştür
        image_data = base64.b64decode(data.split(",")[1])
        image = Image.open(BytesIO(image_data))
        image = np.array(image)

        # Gri tonlamaya çevir
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Canny Kenar Tespit işlemi
        edges = cv2.Canny(image, 100, 200)  # İki threshold değeri ile Canny algoritması

        # İşlenmiş resmi Base64 formatına geri çevir
        _, buffer = cv2.imencode('.png', edges)
        buffer = BytesIO(buffer)
        processed_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({"message": "Canny Edge Detection completed", "image": processed_image_base64})

    except Exception as e:
        return jsonify({"message": f"Bir hata oluştu: {str(e)}"}), 500
    
@app.route('/sobel', methods=['POST'])
def sobel_filter():
    data = request.json.get('image')
    if not data:
        return jsonify({"message": "Resim yüklenmedi!"}), 400

    try:
        # Base64 verisini numpy array'e dönüştür
        image_data = base64.b64decode(data.split(",")[1])
        image = Image.open(BytesIO(image_data))
        image = np.array(image)

        # Gri tonlamaya çevir
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Sobel Filtreleme işlemi (Hem X hem Y yönünde)
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Yatay kenarlar
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Dikey kenarlar

        # Kenarları birleştirme (magnitude)
        sobel_edges = cv2.magnitude(sobel_x, sobel_y)

        # İşlenmiş resmi Base64 formatına geri çevir
        _, buffer = cv2.imencode('.png', sobel_edges)
        buffer = BytesIO(buffer)
        processed_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({"message": "Sobel Edge Detection completed", "image": processed_image_base64})

    except Exception as e:
        return jsonify({"message": f"Bir hata oluştu: {str(e)}"}), 500

@app.route('/laplacian', methods=['POST'])
def laplacian_filter():
    data = request.json.get('image')
    if not data:
        return jsonify({"message": "Resim yüklenmedi!"}), 400

    try:
        # Base64 verisini numpy array'e dönüştür
        image_data = base64.b64decode(data.split(",")[1])
        image = Image.open(BytesIO(image_data))
        image = np.array(image)

        # Gri tonlamaya çevir
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Laplacian Kenar Tespiti
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)  # Sonuçları pozitif değerlere dönüştür

        # İşlenmiş resmi Base64 formatına geri çevir
        _, buffer = cv2.imencode('.png', laplacian)
        buffer = BytesIO(buffer)
        processed_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({"message": "Laplacian Edge Detection completed", "image": processed_image_base64})

    except Exception as e:
        return jsonify({"message": f"Bir hata oluştu: {str(e)}"}), 500

@app.route('/shi-tomasi-corner', methods=['POST'])
def shi_tomasi_corner_detection():
    data = request.json.get('image')
    if not data:
        return jsonify({"message": "Resim yüklenmedi!"}), 400

    try:
        # Base64 verisini numpy array'e dönüştür
        image_data = base64.b64decode(data.split(",")[1])
        image = Image.open(BytesIO(image_data))
        image = np.array(image)

        # Gri tonlamaya çevir
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image

        # Shi-Tomasi köşe tespiti
        corners = cv2.goodFeaturesToTrack(gray_image, 100, 0.01, 10)
        
        if corners is None:
            raise ValueError("Köşe tespiti için yeterli özellik bulunamadı.")  # Köşe tespiti yapılmadıysa hata fırlat

        corners = np.int32(corners)  # np.int0 yerine np.int32 kullanıyoruz

        # Renkli resime dönüştür
        if len(image.shape) == 2:  # Eğer gri tonlama resmi ise, RGB'ye çevir
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Köşe noktalarını kırmızıya boyama
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)  # Kırmızı noktalar

        # İşlenmiş resmi Base64 formatına geri çevir
        _, buffer = cv2.imencode('.png', image)
        buffer = BytesIO(buffer)
        processed_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({"message": "Shi-Tomasi Corner Detection completed", "image": processed_image_base64})

    except Exception as e:
        print(f"Error during Shi-Tomasi Corner Detection: {str(e)}")  # Detaylı hata mesajını yazdır
        return jsonify({"message": f"Bir hata oluştu: {str(e)}"}), 500


    
@app.route('/harris-corner', methods=['POST'])
def harris_corner_detection():
    data = request.json.get('image')
    if not data:
        return jsonify({"message": "Resim yüklenmedi!"}), 400

    try:
        # Base64 verisini kontrol et
        print("Gelen Base64 Resim Verisi:", data)  # Base64 verisini kontrol et

        # Base64 verisini numpy array'e dönüştür
        image_data = base64.b64decode(data.split(",")[1])
        image = Image.open(BytesIO(image_data))
        image = np.array(image)

        # Gri tonlamaya çevir
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image

        # Harris Corner Detection
        dst = cv2.cornerHarris(gray_image, 2, 3, 0.04)

        # Sonuçları görselleştir
        dst = cv2.dilate(dst, None)

        # Renkli resime dönüştür
        if len(image.shape) == 2:  # Eğer gri tonlama resmi ise, RGB'ye çevir
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Harris köşelerini kırmızıya boyama
        image[dst > 0.01 * dst.max()] = [0, 0, 255]  # Köşe noktalarını kırmızı yap

        print("Harris köşe tespiti tamamlandı.")  # Hata ayıklamak için

        # İşlenmiş resmi Base64 formatına geri çevir
        _, buffer = cv2.imencode('.png', image)
        buffer = BytesIO(buffer)
        processed_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        print("Resim Base64 formatına dönüştürüldü.")  # Hata ayıklamak için

        return jsonify({"message": "Harris Corner Detection completed", "image": processed_image_base64})

    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")  # Detaylı hata mesajı
        return jsonify({"message": f"Bir hata oluştu: {str(e)}"}), 500

@app.route('/otsu-thresholding', methods=['POST'])
def otsu_thresholding():
    data = request.json.get('image')
    if not data:
        return jsonify({"message": "Resim yüklenmedi!"}), 400

    try:
        # Base64 verisini numpy array'e dönüştür
        image_data = base64.b64decode(data.split(",")[1])
        image = Image.open(BytesIO(image_data))
        image = np.array(image)

        # Gri tonlamaya çevir
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image

        # Otsu Thresholding işlemi
        _, otsu_thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # İşlenmiş resmi Base64 formatına geri çevir
        _, buffer = cv2.imencode('.png', otsu_thresholded_image)
        buffer = BytesIO(buffer)
        processed_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({"message": "Otsu Thresholding completed", "image": processed_image_base64})

    except Exception as e:
        print(f"Error during Otsu Thresholding: {str(e)}")  # Detaylı hata mesajını yazdır
        return jsonify({"message": f"Bir hata oluştu: {str(e)}"}), 500


@app.route('/video-feed')
def video_feed():
    def generate_frames():
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        camera = cv2.VideoCapture(0)  # Web kamerasını başlat

        while True:
            success, frame = camera.read()
            if not success:
                break

            # Gri tonlamaya çevir
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Yüz algılama
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Yüz etrafına dikdörtgen çizin
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Çerçeveyi JPEG formatına kodla
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Çerçeveyi döndür
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')





if __name__ == '__main__':
    app.run(debug=True)
