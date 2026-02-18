import streamlit as st
import numpy as np
from PIL import Image
import io
import json
import math
from datetime import date

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be FIRST streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Vaidya – AI First Aid",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  html, body, [class*="css"] {
    font-family: 'Segoe UI', Tahoma, sans-serif;
  }
  .main { background: linear-gradient(135deg,#e3f2fd,#ffffff); }

  .vaidya-header {
    background: linear-gradient(90deg,#0d47a1,#1565c0);
    padding: 20px 40px;
    border-radius: 12px;
    margin-bottom: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .vaidya-header h1 { color: white; margin: 0; font-size: 2.2rem; }

  .module-card {
    background: white;
    border-radius: 18px;
    padding: 30px;
    box-shadow: 0 10px 25px rgba(0,0,0,.1);
    text-align: center;
    margin-bottom: 20px;
  }
  .module-card h3 { color: #1565c0; }

  .result-card {
    background: white;
    border-left: 6px solid #1565c0;
    border-radius: 12px;
    padding: 22px 28px;
    margin-top: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,.08);
  }
  .emergency-card { border-left: 6px solid red !important; }
  .result-card h3 { color: #1565c0; }

  .hospital-card {
    background: white;
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 6px 16px rgba(0,0,0,.08);
    border-top: 5px solid #1565c0;
    margin-bottom: 12px;
  }
  .hospital-card h4 { color:#1565c0; margin:0 0 6px 0; }
  .hospital-card p  { margin:3px 0; color:#555; font-size:.9rem; }

  .wound-card {
    background:white;
    border-radius:12px;
    padding:16px 20px;
    box-shadow:0 4px 14px rgba(0,0,0,.07);
    margin-bottom:10px;
  }
  .wound-card h4 { color:#1565c0; margin:0 0 4px 0; }

  .badge-low      { background:#e8f5e9; color:#2e7d32; padding:4px 12px; border-radius:20px; font-size:.85rem; font-weight:600; }
  .badge-medium   { background:#fff3e0; color:#e65100; padding:4px 12px; border-radius:20px; font-size:.85rem; font-weight:600; }
  .badge-high     { background:#ffebee; color:#c62828; padding:4px 12px; border-radius:20px; font-size:.85rem; font-weight:600; }
  .badge-critical { background:#4a148c; color:white;   padding:4px 12px; border-radius:20px; font-size:.85rem; font-weight:600; }

  .confidence-bar {
    background: #e3f2fd;
    border-radius: 10px;
    height: 14px;
    margin: 4px 0 10px 0;
    overflow: hidden;
  }
  .confidence-fill {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #1565c0, #42a5f5);
  }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d47a1,#1565c0);
  }
  section[data-testid="stSidebar"] * { color: white !important; }

  .stButton > button {
    background: linear-gradient(135deg,#1565c0,#0d47a1);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 26px;
    font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  ML MODEL — CLASS LABELS
#
#  ⚠️  YOUR MODEL HAS 7 OUTPUT CLASSES (units=7 in final Dense layer).
#  These class names below must match the folder names you used
#  when training. Edit them if your folder names are different!
# ─────────────────────────────────────────────
CLASS_NAMES = {
    0: "Abrasions",
    1: "Bruises",
    2: "Burns",
    3: "Cut",
    4: "Ingrown_nails",
    5: "Laceration",
    6: "Stab_wound"
}

CLASS_INFO = {
    "Abrasion": {
        "severity": "Low",
        "medication": "Clean with antiseptic, apply antibiotic ointment, cover with bandage",
        "healing": "3–7 days",
        "advice": "Keep the wound clean and dry. Change dressing daily."
    },
    "Bruises": {
        "severity": "Low",
        "medication": "Apply ice pack for 20 minutes, rest the affected area",
        "healing": "1–2 weeks",
        "advice": "Elevate the affected limb if possible to reduce swelling."
    },
    "Burn": {
        "severity": "Medium",
        "medication": "Cool under running water for 10 mins, apply Silver Sulfadiazine Cream, cover loosely",
        "healing": "7–14 days",
        "advice": "Do NOT apply ice, butter or toothpaste. Seek doctor for burns larger than a palm."
    },
    "Cut": {
        "severity": "Medium",
        "medication": "Apply pressure to stop bleeding, clean wound, use butterfly strips or stitches if deep",
        "healing": "5–10 days",
        "advice": "Watch for signs of infection: redness, swelling, or pus."
    },
    "Ingrown_Nail": {
        "severity": "Low",
        "medication": "Soak in warm water, apply antibiotic ointment, wear comfortable footwear",
        "healing": "1–2 weeks",
        "advice": "If severely infected or painful, consult a doctor for minor surgery."
    },
    "Laceration": {
        "severity": "High",
        "medication": "Control bleeding with pressure, clean with saline, requires stitches — go to doctor",
        "healing": "10–20 days",
        "advice": "Lacerations usually require professional medical care. Do not delay."
    },
    "Stab_Wound": {
        "severity": "Critical",
        "medication": "Do NOT remove the object if embedded, apply pressure around wound, call emergency services",
        "healing": "Weeks to months",
        "advice": "CALL AMBULANCE IMMEDIATELY. This is a medical emergency."
    },
}

BADGE = {
    "Low":      "<span class='badge-low'>🟢 Low</span>",
    "Medium":   "<span class='badge-medium'>🟠 Medium</span>",
    "High":     "<span class='badge-high'>🔴 High</span>",
    "Critical": "<span class='badge-critical'>🟣 Critical</span>",
}


# ─────────────────────────────────────────────
#  LOAD MODEL (cached — loads only once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model("first_aid_wound_classifier_7class.h5")
        return model, None
    except ImportError:
        return None, "tensorflow_missing"
    except Exception as e:
        return None, str(e)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize to 224x224, normalize to [0,1], add batch dimension."""
    img = image.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)


def predict_injury(image: Image.Image):
    """Returns: (class_name, confidence_%, all_scores_dict) or (None, error_msg, {})"""

    model, error = load_model()

    if error == "tensorflow_missing":
        return None, "TensorFlow not installed. Run: pip install tensorflow", {}
    if error:
        return None, f"Model error: {error}", {}

    processed = preprocess_image(image)

    # model output shape: (1, 7)
    predictions = model.predict(processed, verbose=0)[0]  # shape becomes (7,)

    # Get predicted class index
    top_idx = int(np.argmax(predictions))

    # Map index to class name
    top_class = CLASS_NAMES[top_idx]

    # Confidence in %
    confidence = float(predictions[top_idx]) * 100

    # Full probability dictionary
    all_scores = {
        CLASS_NAMES[i]: float(predictions[i]) * 100
        for i in range(len(predictions))
    }

    return top_class, confidence, all_scores



# ─────────────────────────────────────────────
#  HOSPITAL DATA
# ─────────────────────────────────────────────
HOSPITALS = [
    {"name":"Apollo Hospitals",           "location":"Jubilee Hills",  "phone":"04023607777","lat":17.4239,"lng":78.4738},
    {"name":"Yashoda Hospitals",          "location":"Somajiguda",     "phone":"04045674567","lat":17.4273,"lng":78.4601},
    {"name":"Care Hospitals",             "location":"Banjara Hills",  "phone":"04030418888","lat":17.4156,"lng":78.4480},
    {"name":"KIMS Hospitals",             "location":"Secunderabad",   "phone":"04044885000","lat":17.4399,"lng":78.4983},
    {"name":"AIG Hospitals",              "location":"Gachibowli",     "phone":"04042444222","lat":17.4418,"lng":78.3636},
    {"name":"Sunshine Hospitals",         "location":"Paradise",       "phone":"04044554455","lat":17.4450,"lng":78.5010},
    {"name":"Continental Hospitals",      "location":"Gachibowli",     "phone":"04067000000","lat":17.4375,"lng":78.3620},
    {"name":"Omega Hospitals",            "location":"Banjara Hills",  "phone":"04023551000","lat":17.4130,"lng":78.4470},
    {"name":"Medicover Hospitals",        "location":"Hitech City",    "phone":"04068334455","lat":17.4489,"lng":78.3714},
    {"name":"Gleneagles Global Hospital", "location":"Lakdi-ka-pul",   "phone":"04030608080","lat":17.3964,"lng":78.4730},
    {"name":"Star Hospitals",             "location":"Banjara Hills",  "phone":"04044777777","lat":17.4140,"lng":78.4475},
    {"name":"Rainbow Children's Hospital","location":"Banjara Hills",  "phone":"04045678900","lat":17.4150,"lng":78.4490},
    {"name":"Osmania General Hospital",   "location":"Afzalgunj",      "phone":"04024600100","lat":17.3810,"lng":78.4720},
    {"name":"Gandhi Hospital",            "location":"Secunderabad",   "phone":"04027505566","lat":17.4420,"lng":78.5000},
    {"name":"NIMS Hospital",              "location":"Punjagutta",     "phone":"04023489000","lat":17.4280,"lng":78.4610},
    {"name":"Basavatarakam Cancer Hosp.", "location":"Banjara Hills",  "phone":"04023551235","lat":17.4200,"lng":78.4500},
    {"name":"Niloufer Hospital",          "location":"Lakdi-ka-pul",   "phone":"04023221234","lat":17.3970,"lng":78.4735},
    {"name":"Virinchi Hospital",          "location":"Banjara Hills",  "phone":"04046999999","lat":17.4160,"lng":78.4485},
    {"name":"MaxCure Hospital",           "location":"Madhapur",       "phone":"04044666666","lat":17.4502,"lng":78.3800},
    {"name":"LV Prasad Eye Institute",    "location":"Banjara Hills",  "phone":"04030612626","lat":17.4180,"lng":78.4458},
    {"name":"Ankura Hospital",            "location":"Madhapur",       "phone":"04044006600","lat":17.4510,"lng":78.3820},
    {"name":"Motherhood Hospital",        "location":"Kondapur",       "phone":"04067229999","lat":17.4600,"lng":78.3720},
    {"name":"Pace Hospitals",             "location":"Hitech City",    "phone":"04048484848","lat":17.4495,"lng":78.3720},
    {"name":"Janani Hospital",            "location":"Kukatpally",     "phone":"04050505050","lat":17.4940,"lng":78.3960},
    {"name":"Life Care Hospital",         "location":"Chandanagar",    "phone":"04055555555","lat":17.4880,"lng":78.3250},
]


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def nearest_hospital(lat, lng):
    return min(HOSPITALS, key=lambda h: haversine(lat, lng, h["lat"], h["lng"]))

def wound_recovery(condition):
    c = condition.lower()
    if "fracture" in c: return "4–8 weeks"
    if "deep"     in c: return "10–20 days"
    if "burn"     in c: return "7–14 days"
    return "3–5 days"


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for key, val in [("users",[]),("wound_data",[]),("logged_in",False),("current_user",None)]:
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("<div class='vaidya-header'><h1>🩺 Vaidya – AI First Aid & Emergency System</h1></div>",
            unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Vaidya")
    st.markdown("---")
    if st.session_state.logged_in:
        st.success(f"👤 {st.session_state.current_user}")
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.rerun()
        st.markdown("---")
    page = st.radio("Navigate to", [
        "🏠 Home","🔐 Login / Sign Up","🏥 Hospital Database",
        "🩹 Wound Tracking","🚨 Emergency System","🤖 AI Injury Detection",
    ])
    st.markdown("---")
    st.caption("© 2026 Vaidya · Developed by Harshitha")


# ═══════════════════════════════════════════════════════════
#  HOME
# ═══════════════════════════════════════════════════════════
if page == "🏠 Home":
    c1, c2 = st.columns([1,1], gap="large")
    with c1:
        st.image("https://img.freepik.com/free-vector/doctor-character-background_1270-84.jpg", width=380)
    with c2:
        st.markdown("## AI-Powered First Aid Assistance")
        st.markdown("Detect wounds with a **real trained AI model**, track healing, and find nearby hospitals instantly.")
        if not st.session_state.logged_in:
            st.info("👆 Use the sidebar to Login / Sign Up and get started!")
        else:
            st.success(f"✅ Welcome, {st.session_state.current_user}!")
    st.markdown("---")
    for col, icon, title, desc in zip(st.columns(4),
        ["🏥","🩹","🚨","🤖"],["Hospitals","Wounds","Emergency","AI Detect"],
        ["Hyderabad directory","Track healing","Find help fast","Real trained model"]):
        with col:
            st.markdown(f"<div class='module-card'><h3>{icon} {title}</h3><p>{desc}</p></div>",
                        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  LOGIN / SIGN UP
# ═══════════════════════════════════════════════════════════
elif page == "🔐 Login / Sign Up":
    st.markdown("## 🔐 Account")
    tab1, tab2 = st.tabs(["Login","Sign Up"])
    with tab1:
        with st.form("lf"):
            em = st.text_input("📧 Email"); pw = st.text_input("🔒 Password", type="password")
            if st.form_submit_button("Login"):
                u = next((u for u in st.session_state.users if u["email"]==em and u["password"]==pw), None)
                if u:
                    st.session_state.logged_in=True; st.session_state.current_user=u["name"]; st.rerun()
                else: st.error("❌ Invalid email or password.")
    with tab2:
        with st.form("sf"):
            nm=st.text_input("👤 Full Name"); em2=st.text_input("📧 Email")
            pw2=st.text_input("🔒 Password",type="password"); cf=st.text_input("🔒 Confirm",type="password")
            if st.form_submit_button("Create Account"):
                if not nm or not em2 or not pw2: st.error("Fill all fields.")
                elif pw2!=cf: st.error("❌ Passwords don't match.")
                elif any(u["email"]==em2 for u in st.session_state.users): st.error("❌ Email already registered.")
                else:
                    st.session_state.users.append({"name":nm,"email":em2,"password":pw2})
                    st.success(f"✅ Account created for {nm}! You can now log in.")


# ═══════════════════════════════════════════════════════════
#  HOSPITAL DATABASE
# ═══════════════════════════════════════════════════════════
elif page == "🏥 Hospital Database":
    st.markdown("## 🏥 Hyderabad Hospital Directory")
    search = st.text_input("🔍 Search", placeholder="e.g. Apollo, Banjara Hills")
    filtered = [h for h in HOSPITALS
                if search.lower() in h["name"].lower() or search.lower() in h["location"].lower()
               ] if search else HOSPITALS
    st.caption(f"**{len(filtered)} hospital(s) found**")
    for i, h in enumerate(filtered):
        with st.columns(3)[i % 3]:
            st.markdown(f"""<div class='hospital-card'>
              <h4>🏥 {h['name']}</h4>
              <p>📍 {h['location']}</p>
              <p>📞 <a href='tel:{h['phone']}'>{h['phone']}</a></p>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  WOUND TRACKING
# ═══════════════════════════════════════════════════════════
elif page == "🩹 Wound Tracking":
    st.markdown("## 🩹 Wound Tracking System")
    with st.form("wf"):
        c1,c2 = st.columns(2)
        with c1: patient=st.text_input("👤 Patient Name")
        with c2: condition=st.text_input("🩺 Condition", placeholder="e.g. burn, deep cut")
        notes=st.text_area("📝 Notes (optional)",height=80)
        if st.form_submit_button("💾 Save Record"):
            if not patient or not condition: st.error("Fill Patient Name and Condition.")
            else:
                rec = wound_recovery(condition)
                st.session_state.wound_data.append({"name":patient,"condition":condition,"notes":notes,"recovery":rec,"date":str(date.today())})
                st.success(f"✅ Saved! Estimated recovery: **{rec}**")
    st.markdown("---")
    st.markdown("### 📋 Saved Records")
    if not st.session_state.wound_data: st.info("No records yet.")
    else:
        for i,w in enumerate(reversed(st.session_state.wound_data)):
            with st.columns(3)[i%3]:
                st.markdown(f"""<div class='wound-card'>
                  <h4>👤 {w['name']}</h4><p>🩺 {w['condition']}</p>
                  <p>⏱ <b>{w['recovery']}</b></p><p>📅 {w['date']}</p>
                  {'<p>📝 '+w['notes']+'</p>' if w['notes'] else ''}
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  EMERGENCY SYSTEM
# ═══════════════════════════════════════════════════════════
elif page == "🚨 Emergency System":
    st.markdown("## 🚨 Emergency System")
    st.warning("⚡ In an emergency — enter your GPS coordinates to find the nearest hospital.")
    st.info("Open Google Maps → long-press your location → copy the coordinates shown.")
    c1,c2 = st.columns(2)
    with c1: lat=st.number_input("🌐 Latitude", value=17.4239,format="%.6f")
    with c2: lng=st.number_input("🌐 Longitude",value=78.4738,format="%.6f")
    if st.button("🚨 Find Nearest Hospital", type="primary"):
        h = nearest_hospital(lat,lng)
        d = haversine(lat,lng,h["lat"],h["lng"])
        st.markdown(f"""<div class='result-card emergency-card'>
          <h3>🚨 Nearest Hospital Found!</h3>
          <p><b>Hospital:</b> {h['name']}</p>
          <p><b>📍 Location:</b> {h['location']}</p>
          <p><b>📞 Contact:</b> <a href='tel:{h['phone']}'>{h['phone']}</a></p>
          <p><b>📏 Distance:</b> ~{d:.1f} km away</p>
          <p><a href='https://www.google.com/maps?q={h["lat"]},{h["lng"]}' target='_blank'>📍 View on Google Maps →</a></p>
        </div>""", unsafe_allow_html=True)
        st.markdown("### 🏥 Top 5 Closest Hospitals")
        for i,h2 in enumerate(sorted(HOSPITALS,key=lambda x:haversine(lat,lng,x["lat"],x["lng"]))[:5]):
            d2=haversine(lat,lng,h2["lat"],h2["lng"])
            st.markdown(f"**{'🥇' if i==0 else '🏥'} {h2['name']}** — {h2['location']} — 📏 {d2:.1f} km — 📞 {h2['phone']}")


# ═══════════════════════════════════════════════════════════
#  AI INJURY DETECTION  ← USES YOUR REAL TRAINED MODEL
# ═══════════════════════════════════════════════════════════
elif page == "🤖 AI Injury Detection":
    st.markdown("## 🤖 AI Injury Detection")
    st.info("Upload a wound photo — your trained MobileNet model will classify it and recommend first aid.")

    with st.expander("ℹ️ What can the AI detect? (7 classes)"):
        cols = st.columns(4)
        for i, cls in enumerate(CLASS_NAMES):
            with cols[i % 4]:
                info = CLASS_INFO.get(cls, {})
                st.markdown(f"**{cls}**")
                st.markdown(BADGE.get(info.get("severity","Low"),""), unsafe_allow_html=True)

    uploaded = st.file_uploader("📁 Upload Injury Image", type=["jpg","jpeg","png","webp"])

    if uploaded:
        col1, col2 = st.columns([1,1], gap="large")
        with col1:
            image = Image.open(uploaded)
            st.image(image, caption="📷 Uploaded Image", use_column_width=True)

        with col2:
            if st.button("🔍 Analyse with AI Model", type="primary"):
                with st.spinner("🧠 Running your trained model…"):
                    predicted_class, confidence, all_scores = predict_injury(image)

                if predicted_class is None:
                    st.error(f"❌ {confidence}")
                    st.code("pip install tensorflow", language="bash")
                else:
                    info     = CLASS_INFO.get(predicted_class, {})
                    severity = info.get("severity","Low")

                    st.markdown(f"""<div class='result-card'>
                      <h3>🩹 Detection Result</h3>
                      <p><b>Injury Type:</b> {predicted_class}</p>
                      <p><b>Severity:</b> {BADGE.get(severity,severity)}</p>
                      <p><b>Confidence:</b> {confidence:.1f}%</p>
                      <p><b>💊 Treatment:</b> {info.get('medication','See a doctor')}</p>
                      <p><b>⏱ Healing Time:</b> {info.get('healing','Varies')}</p>
                      <p><b>💡 Advice:</b> {info.get('advice','Consult a medical professional.')}</p>
                    </div>""", unsafe_allow_html=True)

                    if   severity == "Critical": st.error("🚨 CRITICAL — Call ambulance immediately!")
                    elif severity == "High":     st.warning("⚠️ Needs a doctor soon.")
                    elif severity == "Medium":   st.warning("ℹ️ Monitor carefully.")
                    else:                        st.success("✅ Minor — basic first aid should work.")

                    # ── Confidence bar for all classes ──
                    st.markdown("### 📊 All Class Confidence Scores")
                    for cls_name, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                        color = "#c62828" if cls_name == predicted_class else "#90caf9"
                        st.markdown(f"""
<div style='margin-bottom:6px'>
  <div style='display:flex;justify-content:space-between;font-size:.85rem;'>
    <span>{'✅ ' if cls_name==predicted_class else ''}<b>{cls_name}</b></span>
    <span>{score:.1f}%</span>
  </div>
  <div class='confidence-bar'>
    <div class='confidence-fill' style='width:{int(score)}%;background:{color};'></div>
  </div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style='background:white;border:2px dashed #1565c0;border-radius:16px;
          padding:50px;text-align:center;color:#999;'>
          <h3>📷 Upload an injury image to begin</h3>
          <p>Detects: Abrasion · Bruises · Burn · Cut · Ingrown Nail · Laceration · Stab Wound</p>
        </div>""", unsafe_allow_html=True)
