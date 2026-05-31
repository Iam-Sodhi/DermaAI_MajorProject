import json
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array

# Try multiple GenAI client import styles. Prefer the official `google-genai` package.
genai = None
types = None
try:
    # Preferred import used by the newest Google GenAI client package
    from google import genai as _genai
    from google.genai import types as _types
    genai = _genai
    types = _types
except Exception:
    try:
        # Fallback to older/alternate package name if present
        import google.generativeai as _genai2
        genai = _genai2
        types = getattr(_genai2, "types", None)
    except Exception:
        genai = None
        types = None


# -----------------------------
# App configuration
# -----------------------------
DISEASE_CLASSES = [
    "nevus",
    "melanoma",
    "pigmented benign keratosis",
    "dermatofibroma",
    "squamous cell carcinoma",
    "basal cell carcinoma",
    "vascular lesion",
    "actinic keratosis",
]

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
MODEL_WEIGHTS_PATH = "custom_cnn_skin_disease_classifier_weights.weights.h5"

OFF_TOPIC_RESPONSE = (
    "I can only answer questions about the detected skin condition and directly related "
    "topics such as symptoms, causes, treatment options, prevention, risk factors, "
    "follow-up care, and when to see a dermatologist."
)

MEDICAL_SYSTEM_INSTRUCTION = """
You are a careful dermatology education assistant inside a skin-condition detection app.

The app has detected this possible skin condition: {disease}.

Rules:
1. Answer only questions directly related to the detected condition or closely related skin-health care.
2. If the user asks about anything unrelated, politely refuse and redirect to the detected condition.
3. Do not claim the image model result is a confirmed diagnosis. Say it is an AI prediction and a dermatologist/clinician should confirm it.
4. Give practical, plain-language information about symptoms, causes/risk factors, treatment options, prevention, monitoring, and follow-up when asked.
5. Do not prescribe medication doses or create a personalized treatment plan. Explain common options and recommend professional care.
6. Mention urgent warning signs when relevant, such as rapid growth, bleeding, severe pain, infection signs, or melanoma ABCDE changes.
7. Keep answers concise, accurate, and supportive.
""".strip()

TOPIC_CLASSIFIER_INSTRUCTION = """
You are a strict routing classifier for a dermatology chatbot.

Detected condition: {disease}

Return allowed=true only if the user asks about the detected condition or a directly related skin-health topic, including:
- symptoms, signs, appearance, pain, itching, bleeding, spreading, contagiousness
- causes, risk factors, seriousness, prognosis
- treatment options, prevention, home care, sun protection, monitoring
- diagnosis confirmation, biopsy, dermatologist visit, when to seek urgent care
- medication categories or procedure categories related to the detected skin condition

Return allowed=false for unrelated topics such as programming, schoolwork, sports, celebrities, finance, travel, general trivia, or unrelated medical questions.

Return JSON only in this exact shape:
{{"allowed": true, "reason": "short reason"}}
""".strip()


def get_secret(name: str, default: str = "") -> str:
    """Read a Streamlit secret safely, then fall back to environment variables."""
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return value or os.environ.get(name, default)


def setup_environment_from_secrets() -> None:
    """Load Gemini credentials from Streamlit secrets or environment variables."""
    api_key = get_secret("GOOGLE_API_KEY") or get_secret("GEMINI_API_KEY")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["GEMINI_API_KEY"] = api_key

    model_name = get_secret("GENAI_MODEL")
    if model_name:
        os.environ["GENAI_MODEL"] = model_name


def preprocess_image(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)


@st.cache_resource
def load_model() -> Sequential:
    model = Sequential(
        [
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(224, 224, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(512, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(DISEASE_CLASSES), activation="softmax"),
        ]
    )
    model.load_weights(MODEL_WEIGHTS_PATH)
    return model


@st.cache_resource
def load_llm() -> Dict[str, Any] | None:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    if genai is None:
        return None

    # Newer google-genai exposes a `Client` class (preferred). Older packages
    # may expose a module-level API (configure + functions). Support both
    # where possible; otherwise return None so the app can show a helpful message.
    client = None
    try:
        if hasattr(genai, "Client"):
            client = genai.Client(api_key=api_key)
        elif hasattr(genai, "configure"):
            # Module-style API: configure the module and use it as the client
            genai.configure(api_key=api_key)
            client = genai
    except Exception:
        return None

    if client is None:
        return None

    model_name = os.environ.get("GENAI_MODEL", DEFAULT_GEMINI_MODEL)
    return {"client": client, "model": model_name}


def get_safety_settings() -> List[types.SafetySetting]:
    """Basic harm-category filters for a medical education app."""
    return [
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_MEDIUM_AND_ABOVE",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_MEDIUM_AND_ABOVE",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_MEDIUM_AND_ABOVE",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_ONLY_HIGH",
        ),
    ]


def clean_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON from Gemini, tolerating accidental markdown fences."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    return json.loads(cleaned)


def local_topic_guard(query: str, disease: str) -> Tuple[bool, str]:
    """Conservative fallback if the Gemini classifier is unavailable."""
    q = query.lower()
    disease_terms = {disease.lower(), *disease.lower().split()}
    skin_terms = {
        "skin",
        "lesion",
        "mole",
        "spot",
        "rash",
        "bump",
        "patch",
        "symptom",
        "symptoms",
        "cause",
        "causes",
        "risk",
        "treatment",
        "treat",
        "therapy",
        "medicine",
        "cream",
        "surgery",
        "biopsy",
        "dermatologist",
        "doctor",
        "prevent",
        "prevention",
        "sun",
        "sunscreen",
        "uv",
        "itch",
        "pain",
        "bleed",
        "bleeding",
        "spread",
        "contagious",
        "serious",
        "cancer",
        "benign",
        "malignant",
        "abcde",
        "follow-up",
        "urgent",
    }
    if any(term in q for term in disease_terms | skin_terms):
        return True, "Matched detected-condition or skin-care terms."
    return False, "No detected-condition or skin-care terms found."


def is_on_topic(query: str, disease: str, llm: Dict[str, Any] | None) -> Tuple[bool, str]:
    """Use Gemini as a topic router, with a local keyword fallback."""
    if llm is None:
        return local_topic_guard(query, disease)

    prompt = f"User question: {query}"
    try:
        response = llm["client"].models.generate_content(
            model=llm["model"],
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=TOPIC_CLASSIFIER_INSTRUCTION.format(disease=disease),
                response_mime_type="application/json",
                temperature=0,
                max_output_tokens=120,
                safety_settings=get_safety_settings(),
            ),
        )
        data = clean_json_response(response.text or "{}")
        return bool(data.get("allowed", False)), str(data.get("reason", ""))
    except Exception:
        return local_topic_guard(query, disease)


def format_recent_history(messages: List[Dict[str, str]], max_messages: int = 8) -> str:
    """Keep only recent chat history to avoid oversized prompts."""
    recent = messages[-max_messages:]
    formatted = []
    for message in recent:
        role = message.get("role", "user").upper()
        content = message.get("content", "").strip()
        if role in {"USER", "ASSISTANT"} and content:
            formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


def generate_response(
    query: str,
    disease: str,
    llm: Dict[str, Any] | None,
    chat_history: List[Dict[str, str]],
) -> str:
    """Generate a Gemini answer scoped to the detected disease."""
    if llm is None:
        return (
            "Gemini is not configured. Add your GOOGLE_API_KEY or GEMINI_API_KEY "
            "to Streamlit secrets, then restart the app."
        )

    allowed, _reason = is_on_topic(query, disease, llm)
    if not allowed:
        return f"{OFF_TOPIC_RESPONSE}\n\nDetected condition: **{disease}**."

    history = format_recent_history(chat_history)
    prompt = f"""
Detected condition: {disease}

Recent conversation:
{history if history else "No previous chat messages."}

Answer the latest user question:
{query}
""".strip()

    try:
        response = llm["client"].models.generate_content(
            model=llm["model"],
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=MEDICAL_SYSTEM_INSTRUCTION.format(disease=disease),
                temperature=0.35,
                max_output_tokens=800,
                safety_settings=get_safety_settings(),
            ),
        )
        text = (response.text or "").strip()
        if not text:
            return "I could not generate a safe answer for that question. Please ask about symptoms, treatment, prevention, or follow-up care for the detected condition."
        return text
    except Exception as exc:
        return f"Sorry, I could not contact Gemini right now. Error: {exc}"


def generate_condition_overview(disease: str, llm: Dict[str, Any] | None) -> str:
    """Generate condition information dynamically instead of using hard-coded disease answers."""
    if llm is None:
        return (
            "Gemini is not configured. Add your GOOGLE_API_KEY or GEMINI_API_KEY "
            "to Streamlit secrets to view AI-generated condition information."
        )

    prompt = f"""
Create a concise patient-friendly overview for the detected skin condition: {disease}.
Include these headings only:
- What it is
- Common symptoms/signs
- Usual treatment options
- Prevention and monitoring
- When to see a dermatologist urgently
""".strip()

    try:
        response = llm["client"].models.generate_content(
            model=llm["model"],
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=MEDICAL_SYSTEM_INSTRUCTION.format(disease=disease),
                temperature=0.25,
                max_output_tokens=900,
                safety_settings=get_safety_settings(),
            ),
        )
        return (response.text or "").strip() or "No overview was generated."
    except Exception as exc:
        return f"Could not generate condition information. Error: {exc}"


def reset_chat_for_new_disease(disease: str) -> None:
    if st.session_state.get("chat_disease") != disease:
        st.session_state.messages = []
        st.session_state.chat_disease = disease


def initialize_session_state() -> None:
    defaults = {
        "messages": [],
        "current_disease": None,
        "current_confidence": None,
        "chat_disease": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> str:
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Navigate through the app to explore features:")
    options = ["Upload Image", "View Condition Info", "Chat with Assistant"]
    choice = st.sidebar.radio("Select a section:", options)
    st.sidebar.markdown("___")
    st.sidebar.info("💡 Tip: For better accuracy, upload clear images in good lighting.")
    return choice


def main() -> None:
    setup_environment_from_secrets()
    initialize_session_state()

    choice = render_sidebar()
    st.title("🌟 Skin Disease Detection and Assistant")
    st.caption(
        "Educational tool only. The image model prediction is not a confirmed medical diagnosis. "
        "Please consult a qualified clinician or dermatologist."
    )

    try:
        model = load_model()
    except Exception as exc:
        st.error(f"Could not load the image classifier weights: {exc}")
        return

    llm = load_llm()
    if llm is None:
        st.sidebar.warning("Gemini API key is missing. Add GOOGLE_API_KEY in Streamlit secrets.")
    else:
        st.sidebar.success(f"Gemini model: {llm['model']}")

    if choice == "Upload Image":
        st.subheader("📤 Upload an Image")
        uploaded_file = st.file_uploader(
            "Upload an image of the affected skin area",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            try:
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                predicted_class = DISEASE_CLASSES[int(np.argmax(prediction[0]))]
                confidence = float(np.max(prediction[0]) * 100)

                if st.session_state.current_disease != predicted_class:
                    st.session_state.messages = []
                    st.session_state.chat_disease = predicted_class

                st.session_state.current_disease = predicted_class
                st.session_state.current_confidence = confidence

                st.success(
                    f"Detected Condition: **{predicted_class}** "
                    f"(Model confidence: {confidence:.2f}%)"
                )
                st.info("Go to **Chat with Assistant** to ask questions about this detected condition.")
            except Exception as exc:
                st.error(f"Error processing image: {exc}")

    elif choice == "View Condition Info":
        st.subheader("🔍 Condition Information")
        disease = st.session_state.current_disease
        if not disease:
            st.warning("Upload an image to detect a condition first.")
            return

        st.write(f"**Detected condition:** {disease}")
        if st.session_state.current_confidence is not None:
            st.write(f"**Model confidence:** {st.session_state.current_confidence:.2f}%")

        with st.spinner("Generating condition overview with Gemini..."):
            st.markdown(generate_condition_overview(disease, llm))

    elif choice == "Chat with Assistant":
        st.subheader("💬 Chat with the Assistant")
        disease = st.session_state.current_disease
        if not disease:
            st.warning("Please upload an image to detect a condition before chatting.")
            return

        reset_chat_for_new_disease(disease)
        st.info(
            f"Detected condition: **{disease}**. Ask only about this condition, such as symptoms, "
            "treatment, prevention, risk factors, or follow-up care."
        )

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        chat_prompt = st.chat_input(
            f"Ask about {disease}: symptoms, treatment, prevention, risk, or follow-up..."
        )

        if chat_prompt:
            st.session_state.messages.append({"role": "user", "content": chat_prompt})
            with st.chat_message("user"):
                st.markdown(chat_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Generating answer..."):
                    response = generate_response(
                        query=chat_prompt,
                        disease=disease,
                        llm=llm,
                        chat_history=st.session_state.messages,
                    )
                    st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()