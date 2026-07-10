from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFile
import io
import json
import numpy as np
import cv2
import base64
from models.auth import (authenticate_user, create_user, create_access_token, 
                          decode_token, get_user_by_email)

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = FastAPI(title="MedScan AI")

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/app")
async def serve_app():
    return FileResponse("index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cihazı təyin edirik
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- MODEL YÜKLEME (Universal Funksiya) ----
def load_medical_model(path, num_classes, model_type='b0'):
    """
    Həm EfficientNet-B0, həm də B2 modellərini və fərqli strukturda 
    yadda saxlanılmış (.pth) faylları problemsiz yükləyən universal funksiya.
    """
    # 1. Model tipinə görə arxitekturanı seçirik
    if model_type == 'b2':
        model = models.efficientnet_b2(weights=None)
    else:
        model = models.efficientnet_b0(weights=None)
        
    # 2. Classifier hissəsini təyin edirik
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, num_classes)
    )
    
    # 3. Faylı yükləyirik
    checkpoint = torch.load(path, map_location=device)
    
    # 4. 'KeyError' xətasının qarşısını almaq üçün yoxlama
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        classes = checkpoint['classes']
    else:
        # Əgər fayl birbaşa state_dict-dirsə
        model.load_state_dict(checkpoint)
        # Default olaraq sinif adlarını təyin edirik (əgər faylda yoxdursa)
        classes = ["Sınıq", "Normal"] 
    
    model.eval()
    model = model.to(device)
    return model, classes

# ---- MODELLƏRİN YÜKLƏNMƏSİ ----
print("Modellər yüklənir...")

# Skin disease modeli EfficientNet-B2 olduğu üçün model_type='b2' göndəririk
skin_model, skin_classes = load_medical_model('models/skin_disease_model.pth', 23, model_type='b2')

# Digər bütün modellər EfficientNet-B0-dır (default olaraq b0 işləyəcək)
scoliosis_model, scoliosis_classes = load_medical_model('models/scoliosis_model.pth', 3)
scoliosis_photo_model, scoliosis_photo_classes = load_medical_model('models/scoliosis_photo_model.pth', 2)
chest_model, chest_classes = load_medical_model('models/chest_xray_model.pth', 4)
fracture_model, fracture_classes = load_medical_model('models/bone_fracture_model.pth', 2)
brain_model, brain_classes = load_medical_model('models/brain_tumor_model.pth', 4)

print("Modellər hazırdır!")

# ---- TRANSFORM ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---- PREDİCT FUNKSIYASI ----
def predict(model, image_bytes, classes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = probabilities.argmax().item()
    
    results = {}
    for i, cls in enumerate(classes):
        results[cls] = round(probabilities[i].item() * 100, 2)
    
    return {
        "prediction": classes[predicted_idx],
        "confidence": round(probabilities[predicted_idx].item() * 100, 2),
        "probabilities": results
    }

# ---- GRAD-CAM ----
def generate_gradcam(model, image_bytes, classes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_array = np.array(image.resize((224, 224)))
    
    tensor = transform(image).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    # Forward pass
    model.eval()
    
    # EfficientNet üçün son konvolüsiya qatını tap
    target_layer = model.features[-1]
    
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    handle = target_layer.register_forward_hook(forward_hook)

    outputs = model(tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]
    predicted_idx = probabilities.argmax().item()

    # Backward pass
    model.zero_grad()
    outputs[0, predicted_idx].backward()

    handle.remove()

    # Grad-CAM hesabla
    gradient = gradients[0].cpu().data.numpy()[0]
    activation = activations[0].cpu().data.numpy()[0]

    weights = np.mean(gradient, axis=(1, 2))
    cam = np.zeros(activation.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activation[i]

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    
    cam = cv2.resize(cam, (224, 224))

    # Istilik xəritəsi yarat
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Orijinal şəkil ilə birləşdir
    superimposed = heatmap * 0.4 + img_array * 0.6
    superimposed = np.uint8(superimposed)

    # Base64-ə çevir
    result_image = Image.fromarray(superimposed)
    buffer = io.BytesIO()
    result_image.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return img_base64, predicted_idx, probabilities

# ---- CHEST X-RAY ----
@app.post("/analyze/chest")
@app.post("/analyze/chest")
async def analyze_chest(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # Grad-CAM ilə analiz
    img_base64, predicted_idx, probabilities = generate_gradcam(chest_model, image_bytes, chest_classes)
    
    result = {
        "prediction": chest_classes[predicted_idx],
        "confidence": round(probabilities[predicted_idx].item() * 100, 2),
        "probabilities": {cls: round(probabilities[i].item() * 100, 2) for i, cls in enumerate(chest_classes)},
        "heatmap": img_base64
    }
    
    explanations = {
        "Sağlam": "Ağciyərlərdə əhəmiyyətli patologiya aşkar edilmədi.",
        "Pneumonia": "Pnevmoniya (Ağciyər iltihabı) ehtimalı var. Həkimə müraciət edin.",
        "Cardiomegaly": "Ürək böyüməsi ehtimalı var. Kardioloqa müraciət edin.",
        "Effusion": "Plevral effuziya (ağciyər ətrafında maye) ehtimalı var."
    }
    
    result["explanation"] = explanations.get(result["prediction"], "")
    result["recommendation"] = "Həkimə müraciət edin." if result["prediction"] != "No Finding" else "Nəticələriniz normaldır."
    
    from models.database import save_analysis
    save_analysis(
        analysis_type="chest",
        prediction=result["prediction"],
        confidence=result["confidence"],
        details=str(result["probabilities"])
    )
    return result

# ---- BONE FRACTURE ----
@app.post("/analyze/fracture")
async def analyze_fracture(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    img_base64, predicted_idx, probabilities = generate_gradcam(fracture_model, image_bytes, fracture_classes)
    
    result = {
        "prediction": fracture_classes[predicted_idx],
        "confidence": round(probabilities[predicted_idx].item() * 100, 2),
        "probabilities": {cls: round(probabilities[i].item() * 100, 2) for i, cls in enumerate(fracture_classes)},
        "heatmap": img_base64
    }
    
    explanations = {
        "fractured": "Sümük qırığı ehtimalı yüksəkdir. Təcili ortopedə müraciət edin.",
        "not fractured": "Sümük qırığı aşkar edilmədi."
    }
    
    result["explanation"] = explanations.get(result["prediction"], "")
    result["recommendation"] = "Təcili ortopedə müraciət edin!" if result["prediction"] == "fractured" else "Nəticələriniz normaldır."
    
    from models.database import save_analysis
    save_analysis(
        analysis_type="fracture",
        prediction=result["prediction"],
        confidence=result["confidence"],
        details=str(result["probabilities"])
    )
    return result

# ---- BRAIN TUMOR ----
@app.post("/analyze/brain")
async def analyze_brain(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    img_base64, predicted_idx, probabilities = generate_gradcam(brain_model, image_bytes, brain_classes)
    
    result = {
        "prediction": brain_classes[predicted_idx],
        "confidence": round(probabilities[predicted_idx].item() * 100, 2),
        "probabilities": {cls: round(probabilities[i].item() * 100, 2) for i, cls in enumerate(brain_classes)},
        "heatmap": img_base64
    }
    
    explanations = {
        "glioma": "Glioma şişi ehtimalı var. Təcili neyrologa müraciət edin.",
        "meningioma": "Meningioma şişi ehtimalı var. Neyrologa müraciət edin.",
        "notumor": "Beyin şişi aşkar edilmədi.",
        "pituitary": "Hipofiz şişi ehtimalı var. Endokrinoloqa müraciət edin."
    }
    
    result["explanation"] = explanations.get(result["prediction"], "")
    result["recommendation"] = "Həkimə müraciət edin." if result["prediction"] != "notumor" else "Nəticələriniz normaldır."
    
    from models.database import save_analysis
    save_analysis(
        analysis_type="brain",
        prediction=result["prediction"],
        confidence=result["confidence"],
        details=str(result["probabilities"])
    )
    return result

# ---- QAN ANALİZİ ----
@app.post("/analyze/blood")
async def analyze_blood_route(
    age: int = Form(...),
    gender: str = Form(...),
    hemoglobin: float = Form(...),
    platelet: float = Form(...),
    wbc: float = Form(...),
    rbc: float = Form(...),
    mcv: float = Form(...),
    mch: float = Form(...),
    mchc: float = Form(...)
):
    from models.blood_analyzer import analyze_blood
    result = analyze_blood(age, gender, hemoglobin, platelet, wbc, rbc, mcv, mch, mchc)
    from models.database import save_analysis
    save_analysis(
        analysis_type="blood",
        prediction=result["risk_level"],
        confidence=None,
        details=str(result["findings"])
    )
    return result

# ---- SİDİK ANALİZİ ----
@app.post("/analyze/urine")
async def analyze_urine_route(
    color: str = Form(...),
    clarity: str = Form(...),
    ph: float = Form(...),
    specific_gravity: float = Form(...),
    protein: str = Form(...),
    glucose: str = Form(...),
    ketones: str = Form(...),
    blood: str = Form(...),
    nitrites: str = Form(...),
    leukocytes: str = Form(...)
):
    from models.urine_analyzer import analyze_urine
    result = analyze_urine(color, clarity, ph, specific_gravity, 
                           protein, glucose, ketones, blood, nitrites, leukocytes)
    from models.database import save_analysis
    save_analysis(
        analysis_type="urine",
        prediction=result["risk_level"],
        confidence=None,
        details=str(result["findings"])
    )
    return result

# ---- TİBBİ TERMİN İZAHI ----
@app.get("/explain/{term}")
async def explain_term(term: str):
    from models.medical_terms import get_simple_explanation
    result = get_simple_explanation(term)
    if result:
        return result
    return {"term": term, "simple": "Bu termin haqqında məlumat tapılmadı."}

# ---- QEYDİYYAT ----
@app.post("/auth/register")
async def register(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):
    existing = get_user_by_email(email)
    if existing:
        return {"error": "Bu email artıq qeydiyyatdan keçib"}
    
    if len(password) < 6:
        return {"error": "Şifrə ən az 6 simvol olmalıdır"}
    
    user = create_user(name=name, email=email, password=password)
    token = create_access_token({"sub": user.email, "name": user.name})
    return {
        "success": True,
        "token": token,
        "name": user.name,
        "email": user.email
    }

# ---- GİRİŞ ----
@app.post("/auth/login")
async def login(
    email: str = Form(...),
    password: str = Form(...)
):
    user = authenticate_user(email, password)
    if not user:
        return {"error": "Email və ya şifrə yanlışdır"}
    
    token = create_access_token({"sub": user.email, "name": user.name})
    return {
        "success": True,
        "token": token,
        "name": user.name,
        "email": user.email
    }

# ---- İSTİFADƏÇİ MƏLUMATI ----
@app.get("/auth/me")
async def get_me(authorization: str = None):
    from fastapi import Header
    return {"message": "ok"}

# ---- DƏRİ ANALİZİ ----
@app.post("/analyze/skin")
async def analyze_skin(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    img_base64, predicted_idx, probabilities = generate_gradcam(skin_model, image_bytes, skin_classes)
    
    explanations = {
        "Acne and Rosacea Photos": "Akne və ya rozase əlamətləri var. Dermatoloqa müraciət edin.",
        "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": "Xərçəng xarakterli dəri zədəsi ehtimalı var. Təcili dermatoloqa müraciət edin.",
        "Atopic Dermatitis Photos": "Atopik dermatit (ekzema) əlamətləri var.",
        "Bullous Disease Photos": "Qabarcıqlı dəri xəstəliyi əlamətləri var.",
        "Cellulitis Impetigo and other Bacterial Infections": "Bakterial dəri infeksiyası ehtimalı var.",
        "Eczema Photos": "Ekzema əlamətləri var. Dəri həssaslığı ola bilər.",
        "Exanthems and Drug Eruptions": "Dərman reaksiyası və ya viral töküntü ehtimalı var.",
        "Hair Loss Photos Alopecia and other Hair Diseases": "Saç tökülməsi əlamətləri var.",
        "Herpes HPV and other STDs Photos": "Viral dəri infeksiyası ehtimalı var.",
        "Light Diseases and Disorders of Pigmentation": "Piqmentasiya pozğunluğu əlamətləri var.",
        "Lupus and other Connective Tissue diseases": "Lupus və ya birləşdirici toxuma xəstəliyi ehtimalı var.",
        "Melanoma Skin Cancer Nevi and Moles": "Melanoma (dəri xərçəngi) ehtimalı var. Təcili həkimə müraciət edin!",
        "Nail Fungus and other Nail Disease": "Dırnaq göbələyi və ya xəstəliyi əlamətləri var.",
        "Poison Ivy Photos and other Contact Dermatitis": "Kontakt dermatit əlamətləri var. Allergik reaksiya ola bilər.",
        "Psoriasis pictures Lichen Planus and related diseases": "Psoriaz əlamətləri var. Xroniki dəri xəstəliğidir.",
        "Scabies Lyme Disease and other Infestations and Bites": "Parazit infeksiyası əlamətləri var.",
        "Seborrheic Keratoses and other Benign Tumors": "Xoşxassəli dəri törəməsi əlamətləri var.",
        "Systemic Disease": "Sistematik xəstəliyin dəri əlamətləri var.",
        "Tinea Ringworm Candidiasis and other Fungal Infections": "Göbələk infeksiyası əlamətləri var.",
        "Urticaria Hives": "Ürtiker (kəsəyən) əlamətləri var. Allergik reaksiya ola bilər.",
        "Vascular Tumors": "Damar törəməsi əlamətləri var.",
        "Vasculitis Photos": "Vaskulit (damar iltihabı) əlamətləri var.",
        "Warts Molluscum and other Viral Infections": "Siğil və ya viral dəri infeksiyası əlamətləri var."
    }
    
    prediction = skin_classes[predicted_idx]
    
    result = {
        "prediction": prediction,
        "confidence": round(probabilities[predicted_idx].item() * 100, 2),
        "probabilities": {cls: round(probabilities[i].item() * 100, 2) for i, cls in enumerate(skin_classes)},
        "heatmap": img_base64,
        "explanation": explanations.get(prediction, ""),
        "recommendation": "Təcili dermatoloqa müraciət edin!" if "Malignant" in prediction or "Melanoma" in prediction else "Dermatoloqa müraciət etməyi tövsiyə edirik."
    }
    
    from models.database import save_analysis
    save_analysis(
        analysis_type="skin",
        prediction=result["prediction"],
        confidence=result["confidence"],
        details=str(result["probabilities"])
    )
    return result

# ---- SKALİOZ ANALİZİ (Rentgen) ----
@app.post("/analyze/scoliosis")
async def analyze_scoliosis(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_base64, predicted_idx, probabilities = generate_gradcam(scoliosis_model, image_bytes, scoliosis_classes)
    
    explanations = {
        "Normal": "Onurğa sütununda skalioz əlaməti aşkar edilmədi. Vəziyyətiniz normaldır.",
        "Scol": "Skalioz əlaməti aşkar edildi. Onurğa sütununuzda əyrilik var. Ortopedə müraciət edin.",
        "Spond": "Spondylolisthesis əlaməti aşkar edildi. Onurğa fəqərələrindən biri öz yerindən sürüşüb. Həkimə müraciət edin."
    }
    
    prediction = scoliosis_classes[predicted_idx]
    result = {
        "prediction": prediction,
        "confidence": round(probabilities[predicted_idx].item() * 100, 2),
        "probabilities": {cls: round(probabilities[i].item() * 100, 2) for i, cls in enumerate(scoliosis_classes)},
        "heatmap": img_base64,
        "explanation": explanations.get(prediction, ""),
        "recommendation": "Təcili ortopedə müraciət edin!" if prediction != "Normal" else "Nəticələriniz normaldır."
    }
    
    from models.database import save_analysis
    save_analysis(analysis_type="scoliosis", prediction=result["prediction"],
                  confidence=result["confidence"], details=str(result["probabilities"]))
    return result

# ---- SKALİOZ FOTO ANALİZİ ----
@app.post("/analyze/scoliosis_photo")
async def analyze_scoliosis_photo(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_base64, predicted_idx, probabilities = generate_gradcam(scoliosis_photo_model, image_bytes, scoliosis_photo_classes)
    
    explanations = {
        "Normal": "Foto analizinə əsasən skalioz riski aşkar edilmədi. Vəziyyətiniz normaldır.",
        "Skolioz": "Foto analizinə əsasən skalioz riski var. Dəqiq diaqnoz üçün rentgen çəkdirin və ortopedə müraciət edin."
    }
    
    prediction = scoliosis_photo_classes[predicted_idx]
    result = {
        "prediction": prediction,
        "confidence": round(probabilities[predicted_idx].item() * 100, 2),
        "probabilities": {cls: round(probabilities[i].item() * 100, 2) for i, cls in enumerate(scoliosis_photo_classes)},
        "heatmap": img_base64,
        "explanation": explanations.get(prediction, ""),
        "recommendation": "Rentgen çəkdirin və ortopedə müraciət edin!" if prediction == "scoliosis" else "Nəticələriniz normaldır."
    }
    
    from models.database import save_analysis
    save_analysis(analysis_type="scoliosis_photo", prediction=result["prediction"],
                  confidence=result["confidence"], details=str(result["probabilities"]))
    return result

# ---- ANA SƏHIFƏ ----
@app.get("/")
async def root():
    return {"message": "MedScan AI API işləyir!", "version": "1.0"}

# ---- AI DOCTOR CHAT ----
@app.post("/chat")
async def chat(message: str = Form(...)):
    from models.chat_engine import get_response
    response = get_response(message)
    return {"response": response}

# ---- TARİX ----
@app.get("/history")
async def get_analysis_history():
    from models.database import get_history
    history = get_history()
    return {"history": history}

# ---- STATİSTİKA ----
@app.get("/stats")
async def get_analysis_stats():
    from models.database import get_stats
    stats = get_stats()
    return stats

# ---- TARİXİ TƏMİZLƏ ----
@app.delete("/history/clear")
async def clear_history():
    from models.database import clear_all_history
    clear_all_history()
    return {"message": "Tarixçə təmizləndi"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)