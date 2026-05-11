import streamlit as st
import requests
from transformers import pipeline
import re

# Page config
st.set_page_config(page_title="AI Healthcare Assistant", page_icon="🤖🏥", layout="wide")

# Built-in comprehensive medical knowledge base
MEDICAL_KNOWLEDGE = {
    "diabetes": "Diabetes is a chronic condition affecting blood glucose regulation. Type 1: autoimmune, pancreas doesn't produce insulin. Type 2: insulin resistance, most common form. Gestational: occurs during pregnancy. Symptoms: thirst, frequent urination, fatigue, blurred vision, slow wound healing. Risks: obesity, family history, age. Management: diet, exercise, medication, blood sugar monitoring. Complications if untreated: kidney damage, neuropathy, cardiovascular disease, blindness.",
    
    "hypertension": "High blood pressure (systolic >130 or diastolic >80 mmHg) strains arteries. Called 'silent killer' - often no symptoms until damage occurs. Primary: no identifiable cause (90%). Secondary: from kidney disease, hormonal disorders. Risk factors: obesity, stress, high salt, alcohol, sedentary lifestyle, family history. Diagnosis: blood pressure test. Complications: heart attack, stroke, kidney disease, heart failure. Management: lifestyle changes (diet, exercise, stress reduction), medication (ACE inhibitors, beta-blockers, diuretics), regular monitoring.",
    
    "asthma": "Chronic inflammatory airway disease causing breathing difficulty. Airways narrow, produce excess mucus, inflame. Symptoms: wheezing, shortness of breath, chest tightness, cough (especially at night/exercise). Triggers: allergens (pollen, dust, pets), smoke, air pollution, exercise, cold air, strong odors, stress, respiratory infections. Types: allergic, non-allergic, occupational, exercise-induced. Management: inhalers (quick-relief, maintenance), avoiding triggers, peak flow monitoring. Severe attacks require emergency care.",
    
    "fever": "Elevated body temperature (>100.4°F/38°C) indicating infection fight. Body raises temperature to kill pathogens. Symptoms: chills, body aches, fatigue, headache, dehydration. Common causes: viral (cold, flu), bacterial (strep, UTI), fungal infections. Mild fever is beneficial. Seek doctor if: very high (>104°F), lasts >3 days, in young children/elderly, with severe symptoms. Treatment: rest, fluids, acetaminophen/ibuprofen. Cool compresses help comfort.",
    
    "headache": "Pain in head/scalp region. Types: Tension (pressure, 70% of headaches, stress-related), Migraine (throbbing, unilateral, with aura/nausea, 15%), Cluster (severe eye pain, recurring cycles, 1%), Other (medication overuse, hormonal, dehydration). Triggers: stress, sleep deprivation, caffeine, alcohol, hormonal changes, poor posture, bright lights. Management: rest, hydration, over-the-counter pain relievers, massage, dark quiet room. See doctor if: sudden severe, frequent, with neurological symptoms.",
    
    "cold": "Common viral infection affecting upper respiratory tract. Caused by rhinovirus, coronavirus, others. Symptoms: runny/stuffy nose, sore throat, cough, sneezing, mild fatigue, low-grade fever. Incubation: 1-3 days. Duration: 7-10 days typically. Contagious: 2-3 days before to 8 days after symptoms. Prevention: hand washing, distance from sick people. Treatment: rest, fluids, saline nasal drops, honey for cough, pain relievers for aches. Antibiotics don't help (viral).",
    
    "flu": "Influenza is contagious viral respiratory illness. More severe than cold. Symptoms: sudden fever (100-104°F), body aches, chills, fatigue, cough, sore throat. Onset: 1-4 days after exposure. Risk groups: elderly, young children, pregnant women, immunocompromised. Complications: pneumonia, bronchitis. Prevention: annual flu vaccine (60-80% effective). Treatment: antiviral medication (oseltamivir) if started early, rest, fluids, pain relievers. Isolation: first 5-7 days.",
    
    "migraine": "Neurological condition causing intense throbbing head pain (usually one side). With aura: visual disturbances, sensory symptoms before headache. Triggers: stress, hormonal changes, certain foods (aged cheese, MSG, caffeine), sleep changes, weather changes, bright lights. Symptoms: nausea, vomiting, light/sound sensitivity, lasting 4-72 hours. Prevention: identify triggers, stress management, regular sleep, hydration. Acute treatment: NSAIDs, triptans, ergotamines. Preventive: beta-blockers, antidepressants, anticonvulsants for frequent migraines.",
    
    "anxiety": "Mental health disorder with excessive worry, fear. Symptoms: nervousness, rapid heartbeat, sweating, trembling, difficulty concentrating, sleep issues, muscle tension. Types: generalized anxiety disorder (GAD), panic disorder, social anxiety, specific phobias. Triggers: stress, caffeine, lack of sleep, health concerns. Management: cognitive behavioral therapy (CBT), meditation, exercise, deep breathing, limiting caffeine. Medication: SSRIs (selective serotonin reuptake inhibitors), anti-anxiety drugs. Crisis: breathing exercises (4-4-4-4 count).",
    
    "depression": "Mental health condition affecting mood, motivation, energy. Symptoms: persistent sadness, loss of interest, fatigue, sleep changes (insomnia or hypersomnia), appetite changes, difficulty concentrating, worthlessness, suicidal thoughts. Duration: at least 2 weeks for diagnosis. Causes: chemical imbalances, life events, genetics, chronic illness. Treatment: psychotherapy (CBT, psychodynamic), antidepressants (SSRIs, SNRIs), lifestyle (exercise, sleep, sunlight). Emergency: if suicidal, call crisis line immediately.",
    
    "arthritis": "Joint inflammation causing pain, stiffness. Osteoarthritis: wear-and-tear, age-related, cartilage breakdown. Rheumatoid: autoimmune, affects multiple joints symmetrically. Symptoms: joint pain, swelling, reduced range of motion, stiffness (worse morning), fatigue (RA). Risk: age, obesity, previous injury, family history. Management: exercise, weight management, hot/cold therapy, NSAIDs, corticosteroids. Advanced: joint replacement surgery. Prevention: stay active, maintain weight.",
    
    "high cholesterol": "Elevated cholesterol (>200 mg/dL total) increases heart disease/stroke risk. LDL ('bad') deposits in arteries. HDL ('good') removes cholesterol. Triglycerides another risk factor. Often asymptomatic. Risk: genetics, age, diet, sedentary, obesity, smoking, diabetes. Prevention: heart-healthy diet (soluble fiber, omega-3, limit saturated fat), exercise 30min 5x/week, weight management, quit smoking. Treatment: statins (most common), ezetimibe, PCSK9 inhibitors. Goal: LDL <100 mg/dL.",
    
    "gastritis": "Stomach lining inflammation. Causes: H. pylori bacteria, NSAIDs, stress, alcohol, smoking, autoimmune. Acute: sudden onset. Chronic: long-term. Symptoms: upper abdominal pain/discomfort, nausea, vomiting, loss of appetite, bloating, belching, dark stool. Diagnosis: endoscopy, breath/stool tests. Treatment: acid reducers (H2 blockers, PPIs), antibiotics if H. pylori, avoiding triggers. Severe: can lead to ulcers, bleeding.",
    
    "insomnia": "Persistent difficulty sleeping (falling asleep, staying asleep, early awakening). Acute: short-term due to stress/life events. Chronic: >3 months, occurs 3+ nights/week. Causes: stress, anxiety, depression, poor sleep habits, caffeine, irregular schedule, medical conditions, age, hormones. Symptoms: daytime fatigue, irritability, difficulty concentrating. Treatment: sleep hygiene (consistent schedule, dark room, cool temp, no screens 1hr before), relaxation techniques, CBT-I (cognitive behavioral therapy), melatonin, prescription sleep aids if severe.",
    
    "obesity": "Excess body fat (BMI >30). Health risks: diabetes, heart disease, hypertension, stroke, sleep apnea, fatty liver, certain cancers, joint problems. Causes: genetics (25-70%), overeating, sedentary lifestyle, medications, medical conditions (hypothyroidism, PCOS), psychological factors. Assessment: BMI, waist circumference, body composition. Management: balanced diet (calorie deficit), regular exercise (150min/week aerobic + strength), behavioral changes, stress management, sleep. Medication/surgery for severe cases.",
    
    "allergies": "Immune system overreaction to harmless substances (allergens). Common: pollen, dust mites, pet dander, food (peanuts, shellfish), mold, bee venom. Symptoms: sneezing, itchy/watery eyes, runny/stuffy nose, itchy throat/ears, skin rash, anaphylaxis (severe). Seasonal vs perennial. Diagnosis: skin prick test, blood tests. Prevention: avoid triggers, air filters, wash bedding. Treatment: antihistamines, decongestants, nasal steroids, immunotherapy. Emergency: epinephrine for anaphylaxis.",
    
    "bronchitis": "Airway inflammation producing cough. Acute: viral (post-cold/flu), 1-3 weeks. Chronic: >3 months cough, smoking-related (COPD). Symptoms: persistent cough (dry then phlegmy), chest discomfort, fatigue, shortness of breath, mild fever. Diagnosis: chest X-ray if concerning. Treatment: rest, fluids, humidifier, cough medicine (guaifenesin), bronchodilators if wheezing. Antibiotics: only if bacterial. Quit smoking to prevent chronic.",
    
    "pneumonia": "Lung infection causing air sacs (alveoli) to fill with fluid/pus. Bacterial, viral, or fungal. Symptoms: cough (productive with yellow/green sputum), fever, chills, chest pain, shortness of breath, fatigue. Diagnosis: chest X-ray, sputum culture. Severity: community-acquired (mild-moderate) vs hospital-acquired (severe). Treatment: antibiotics (bacterial), antivirals (viral), supportive care (oxygen, fluids). Vaccines prevent some types. Emergency if: severe symptoms, low oxygen.",
    
    "urinary tract infection": "Infection of bladder (cystitis) or urethra (urethritis). More common in women. Symptoms: burning urination, frequent urge, urgency, cloudy/bloody urine, lower abdominal pain, pelvic pressure. Causes: bacteria (E. coli), catheter use, incomplete bladder emptying, sexual activity. Diagnosis: urinalysis, urine culture. Prevention: hydration, wiping front-to-back, urinate after intercourse, avoid irritants. Treatment: antibiotics (trimethoprim-sulfamethoxazole, fluoroquinolone), urinary analgesics for pain.",
    
    "constipation": "Infrequent or difficult bowel movements (<3/week, hard stools). Causes: low fiber/water intake, sedentary lifestyle, medications (opioids, antacids), irritable bowel syndrome, hypothyroidism, age. Symptoms: straining, incomplete evacuation, abdominal discomfort, bloating. Acute: sudden onset. Chronic: long-standing. Prevention: fiber 25-35g/day, 8 glasses water/day, exercise, regular schedule. Treatment: stool softeners, laxatives (osmotic, stimulant), suppositories. Severe: impaction needs medical care.",
    
    "diarrhea": "Frequent loose/watery stools (>3/day). Acute: bacterial/viral infection (food poisoning, gastroenteritis), medication side effects. Chronic: IBS, inflammatory bowel disease, malabsorption, lactose intolerance. Risks: dehydration (electrolyte loss), especially children/elderly. Symptoms: urgency, cramping, fever if infectious. Treatment: hydration (oral rehydration solutions), bland diet (BRAT), avoid dairy/fiber initially, bismuth salts (Pepto), loperamide (Imodium). Antibiotics only if bacterial. Seek care if bloody, severe, prolonged.",
    
    "iron deficiency anemia": "Low red blood cells/hemoglobin from iron deficiency. Symptoms: fatigue, weakness, shortness of breath, pale skin, dizziness, cold extremities, brittle nails. Causes: poor diet, blood loss (heavy periods, GI bleeding), pregnancy, malabsorption. Diagnosis: blood test (hemoglobin, ferritin, iron levels). Risk: women, vegetarians, young children, elderly. Treatment: iron supplements (ferrous sulfate), dietary iron (red meat, spinach, beans), treat underlying cause. Severe: blood transfusion if acute.",
    
    "vitamin d deficiency": "Inadequate vitamin D (>30 ng/mL normal). Causes: limited sun exposure, poor diet, dark skin in high latitudes, malabsorption. Symptoms: fatigue, bone/muscle pain, weakness, mood changes, frequent infections. Complications: osteoporosis, increased fracture risk, autoimmune disease risk. Diagnosis: blood test (25-hydroxyvitamin D). Prevention: sun exposure 10-30min daily, fatty fish, supplements. Treatment: vitamin D3 supplements, fortified foods, sunlight. Deficiency increases cancer, cardiovascular disease, infection risks.",
    
    "blood pressure": "Force blood exerts on artery walls. Normal: <120/<80 mmHg. Elevated: 120-129/<80. Stage 1 hypertension: 130-139/80-89. Stage 2: >140/>90. Measurement: systolic (pressure during heartbeat) over diastolic (between beats). Home monitoring recommended for diagnosis confirmation. Regular checks important - many people unaware of high BP. Lifestyle changes first line: exercise, diet (DASH diet), weight, stress, limit salt/alcohol, quit smoking. Medication if lifestyle insufficient.",
    
    "sleep health": "Quality sleep (7-9 hours adults) essential for immunity, metabolism, mood, memory, cardiovascular health. Sleep stages: light, deep (restorative), REM (dreaming). Circadian rhythm influenced by light, body temp, hormones. Good sleep hygiene: consistent schedule, dark/cool room (65-68°F), no screens 1hr before, avoid caffeine after 2pm, exercise earlier in day, relax before bed (meditation, reading). Common issues: insomnia, sleep apnea, restless legs. Chronic poor sleep increases disease risk.",
    
    "heart disease": "Broad term for conditions affecting heart. Coronary artery disease (CAD): plaque buildup narrows arteries. Symptoms: chest pain (angina), shortness of breath, fatigue, irregular heartbeat. Risk factors: high cholesterol, hypertension, smoking, diabetes, obesity, family history, sedentary lifestyle. Diagnosis: ECG, stress test, echocardiogram, cardiac catheterization. Management: lifestyle changes (diet, exercise, quit smoking), medications (statins, beta-blockers, aspirin), procedures (angioplasty, bypass surgery). Prevention: healthy diet, regular exercise, weight management.",
    
    "stroke": "Sudden brain damage from interrupted blood flow. Ischemic (87%): clot blocks artery. Hemorrhagic: bleeding in brain. Symptoms: sudden numbness/weakness (face/arm/leg), confusion, trouble speaking/seeing, difficulty walking, severe headache. Risk factors: hypertension, smoking, diabetes, high cholesterol, atrial fibrillation, obesity, family history. Diagnosis: CT/MRI scan, blood tests. Treatment: clot-busting drugs (tPA) for ischemic, surgery for hemorrhagic. Rehabilitation: physical/occupational/speech therapy. Prevention: control risk factors, healthy lifestyle.",
    
    "eczema": "Chronic skin condition causing inflammation, redness, itching. Atopic dermatitis most common. Symptoms: dry/itchy skin, red patches, small bumps, thickened skin from scratching. Triggers: allergens (dust, pollen), irritants (soaps, wool), stress, weather changes, infections. Risk: family history, asthma/allergies. Diagnosis: clinical exam, skin biopsy if needed. Management: moisturizers, topical steroids, antihistamines, avoid triggers. Severe: immunosuppressants, phototherapy. Prevention: gentle skincare, humidifiers, stress management.",
    
    "back pain": "Pain in back region, from neck to lower back. Acute: sudden onset, resolves in weeks. Chronic: persists >3 months. Causes: muscle strain, herniated disc, arthritis, osteoporosis, poor posture, injury. Symptoms: dull ache to sharp pain, stiffness, limited mobility, numbness/tingling if nerve compression. Diagnosis: physical exam, X-ray, MRI if severe. Management: rest, ice/heat, over-the-counter pain relievers, physical therapy, exercise. Prevention: good posture, regular exercise, proper lifting techniques.",
    
    "kidney stones": "Hard mineral deposits in kidneys. Types: calcium oxalate (most common), uric acid, struvite, cystine. Symptoms: severe flank pain, blood in urine, nausea, vomiting, frequent urination. Risk factors: dehydration, high salt/protein diet, obesity, family history, certain medications. Diagnosis: ultrasound, CT scan, urinalysis. Treatment: pain medication, hydration to pass stone, medications to relax ureter, surgery for large stones. Prevention: drink plenty water, reduce salt, limit oxalate foods (spinach, rhubarb)."
}

# ✅ Better model (Transformers 4.40 compatible)
@st.cache_resource
def load_model():
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return generator, summarizer

model, summarizer = load_model()

# Fetch medical context (built-in + API fallback)
def fetch_medical_context(query: str) -> str:
    query_lower = query.lower()
    
    # Keyword synonyms for better matching
    synonyms = {
        "diabetes": ["glucose", "blood sugar", "type 1", "type 2", "diabetic"],
        "hypertension": ["high blood pressure", "bp", "blood pressure", "elevated pressure"],
        "asthma": ["respiratory", "breathing", "airways", "wheezing"],
        "fever": ["temperature", "high temp", "hot", "feverish"],
        "headache": ["head pain", "migraine", "tension", "cluster"],
        "cold": ["common cold", "rhinovirus", "runny nose", "nasal", "congestion"],
        "flu": ["influenza", "viral", "respiratory illness"],
        "migraine": ["throbbing", "neurological", "aura"],
        "anxiety": ["anxious", "worry", "panic", "nervousness", "stress disorder"],
        "depression": ["depressed", "sad", "mood disorder", "mental health"],
        "arthritis": ["joint pain", "inflammation", "osteoarthritis", "rheumatoid"],
        "cholesterol": ["cholesterol", "lipid", "ldl", "hdl"],
        "gastritis": ["stomach", "gastric", "ulcer", "acid"],
        "insomnia": ["sleep", "sleeping", "can't sleep", "sleepless"],
        "obesity": ["overweight", "excess weight", "bmi"],
        "allergy": ["allergic", "allergen", "allergies"],
        "bronchitis": ["bronchial", "cough", "airways"],
        "pneumonia": ["lung infection", "lung", "alveoli"],
        "uti": ["urinary", "uti", "bladder", "urine"],
        "constipation": ["bowel", "stool", "digestion"],
        "diarrhea": ["loose stool", "digestive", "intestinal"],
        "anemia": ["low iron", "hemoglobin", "blood count"],
        "vitamin d": ["vitamin d", "deficiency", "sun exposure"],
        "blood pressure": ["bp", "pressure", "systolic", "diastolic"],
        "sleep": ["sleep", "sleeping", "rest", "insomnia", "sleep quality"]
    }
    
    # Check built-in knowledge base with synonym matching
    for key, knowledge in MEDICAL_KNOWLEDGE.items():
        # Direct match
        if key in query_lower:
            return knowledge
        # Synonym matching
        if key in synonyms:
            for synonym in synonyms[key]:
                if synonym in query_lower:
                    return knowledge
    
    # Fallback to Google API if not in knowledge base
    try:
        GOOGLE_API_KEY = "AIzaSyBxphvqMeyttf8GOOKfYEgcMBH00bHnT0A"
        SEARCH_ENGINE_ID = "91c0096f6f54c48db"

        search_url = f"https://www.googleapis.com/customsearch/v1?q={query}+medical+health&cx={SEARCH_ENGINE_ID}&key={GOOGLE_API_KEY}"
        response = requests.get(search_url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            if "items" in data:
                item = data["items"][0]
                return f"{item['title']}. {item['snippet']}"
    except Exception as e:
        pass
    
    return "General medical information."

# -------------------------------
# Clean output
# -------------------------------
def clean_response(response: str):
    # Remove common filler patterns first
    unwanted_phrases = [
        "As an AI language model",
        "I'm here to help",
        "However",
        "It is important to note",
        "Medical Question:",
        "Medical Context:",
        "Answer:",
        "diabetes is a chronic",
        "Give a complete",
        "Provide a comprehensive",
        "Give a complete description",
    ]
    for phrase in unwanted_phrases:
        response = response.replace(phrase, '')
    
    # Remove duplicate words
    response = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', response, flags=re.IGNORECASE)
    
    # Clean up whitespace
    response = re.sub(r'\n+', ' ', response).strip()
    response = re.sub(r'\s+', ' ', response)
    
    # Remove duplicate sentences
    sentences = [s.strip() for s in response.split('. ') if s.strip()]
    seen = set()
    unique_sentences = []
    for sent in sentences:
        # Normalize for comparison (remove punctuation)
        normalized = re.sub(r'[.,!?]', '', sent.lower()).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_sentences.append(sent)
    
    response = '. '.join(unique_sentences)
    if response and not response.endswith('.'):
        response += '.'
    
    # Clean up any stray punctuation
    response = re.sub(r'\s+([,.!?])', r'\1', response)
    response = re.sub(r'^[,.!?\s]+', '', response)  # Remove leading punctuation
    
    return response.strip()

# -------------------------------
# Medicine API (UNCHANGED)
# -------------------------------
def get_medicine_info(medicine_name):
    url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{medicine_name}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            drug_info = data["results"][0]

            return {
                "Brand Name": drug_info["openfda"].get("brand_name", ["N/A"])[0],
                "Generic Name": drug_info["openfda"].get("generic_name", ["N/A"])[0],
                "Manufacturer": drug_info["openfda"].get("manufacturer_name", ["N/A"])[0],
                "Purpose": drug_info.get("purpose", ["N/A"])[0],
                "Warnings": drug_info.get("warnings", ["N/A"])[0],
                "Dosage": drug_info.get("dosage_and_administration", ["N/A"])[0],
                "Side Effects": drug_info.get("adverse_reactions", ["N/A"])[0],
                "Interactions": drug_info.get("drug_interactions", ["N/A"])[0],
            }
    return None

def get_query_intent(query: str) -> str:
    q = query.lower()
    if any(phrase in q for phrase in ["what triggers", "triggers", "triggered by", "trigger", "what causes", "caused by"]):
        return "Triggers"
    if any(phrase in q for phrase in ["symptom", "symptoms", "how do i know", "signs", "present", "what are the signs", "what are the symptoms"]):
        return "Symptoms"
    if any(phrase in q for phrase in ["manage", "treatment", "treat", "therapy", "how to treat", "what should i do", "control"]):
        return "Management"
    if any(phrase in q for phrase in ["diagnose", "diagnosis", "test", "how do i know", "identify", "find out", "have"]):
        return "Diagnosis"
    if any(phrase in q for phrase in ["what is", "define", "meaning of", "explain"]):
        return "Definition"
    return "Definition"


def extract_kb_section(article: str, section: str) -> str | None:
    section = section.capitalize()
    if section == "Definition":
        return article
    pattern = rf"{section}:(.*?)(?:[A-Z][a-z ]+:|$)"
    match = re.search(pattern, article, flags=re.IGNORECASE | re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        if extracted:
            return re.sub(r'\s+', ' ', extracted)
    return None


def format_kb_answer(condition: str, intent: str, content: str) -> str:
    text = content.strip().rstrip('.')
    condition_name = condition.capitalize()
    if intent == "Triggers":
        return f"{condition_name} triggers include {text}."
    if intent == "Symptoms":
        return f"Common symptoms of {condition_name} include {text}."
    if intent == "Management":
        return f"To manage {condition_name}, {text}."
    if intent == "Diagnosis":
        return f"Diagnosis of {condition_name} may involve {text}."
    return content


# Helper function to find knowledge base answer with synonym matching
def find_kb_answer(query: str):
    query_lower = query.lower()
    
    synonyms = {
        "diabetes": ["glucose", "blood sugar", "type 1", "type 2", "diabetic"],
        "hypertension": ["high blood pressure", "bp", "blood pressure", "elevated pressure"],
        "asthma": ["respiratory", "breathing", "airways", "wheezing"],
        "fever": ["temperature", "high temp", "hot", "feverish"],
        "headache": ["head pain", "migraine", "tension", "cluster"],
        "cold": ["common cold", "rhinovirus", "runny nose", "nasal", "congestion"],
        "flu": ["influenza", "viral", "respiratory illness"],
        "migraine": ["throbbing", "neurological", "aura"],
        "anxiety": ["anxious", "worry", "panic", "nervousness", "stress disorder"],
        "depression": ["depressed", "sad", "mood disorder", "mental health"],
        "arthritis": ["joint pain", "inflammation", "osteoarthritis", "rheumatoid"],
        "cholesterol": ["cholesterol", "lipid", "ldl", "hdl"],
        "gastritis": ["stomach", "gastric", "ulcer", "acid"],
        "insomnia": ["sleep", "sleeping", "can't sleep", "sleepless"],
        "obesity": ["overweight", "excess weight", "bmi"],
        "allergy": ["allergic", "allergen", "allergies"],
        "bronchitis": ["bronchial", "cough", "airways"],
        "pneumonia": ["lung infection", "lung", "alveoli"],
        "uti": ["urinary", "uti", "bladder", "urine"],
        "constipation": ["bowel", "stool", "digestion"],
        "diarrhea": ["loose stool", "digestive", "intestinal"],
        "anemia": ["low iron", "hemoglobin", "blood count"],
        "vitamin d": ["vitamin d", "deficiency", "sun exposure"],
        "blood pressure": ["bp", "pressure", "systolic", "diastolic"],
        "sleep": ["sleep", "sleeping", "rest", "insomnia", "sleep quality"]
    }
    
    # Check built-in knowledge base with synonym matching
    for key, knowledge in MEDICAL_KNOWLEDGE.items():
        if key in query_lower:
            intent = get_query_intent(query)
            section = extract_kb_section(knowledge, intent)
            return format_kb_answer(key, intent, section) if section else knowledge
        if key in synonyms:
            for synonym in synonyms[key]:
                if synonym in query_lower:
                    intent = get_query_intent(query)
                    section = extract_kb_section(knowledge, intent)
                    return format_kb_answer(key, intent, section) if section else knowledge
    return None

# -------------------------------
# UI Layout
# -------------------------------
col1, col2 = st.columns([2, 1])

# -------------------------------
# LEFT SIDE (Main Chat)
# -------------------------------
with col1:
    st.title("🤖🏥 MediBot - AI Healthcare Assistant")

    st.warning("⚠️ This AI assistant is for informational purposes only and not a substitute for professional medical advice.")

    user_input = st.text_area(
        "Enter your medical question:",
        placeholder="e.g., What are the symptoms of diabetes?"
    )

    if st.button("Get Answer"):
        if user_input:

            with st.spinner("Fetching medical context..."):
                context = fetch_medical_context(user_input)

            if not context:
                context = "General medical knowledge."

            with st.spinner("Generating response..."):
                try:
                    # ✅ Check if we have direct knowledge base match (with synonym matching)
                    direct_answer = find_kb_answer(user_input)
                    
                    if direct_answer:
                        # Use knowledge base answer directly - it's cleaner and avoids repetition
                        answer = direct_answer
                        from_knowledge_base = True
                    else:
                        # Use model for questions not in knowledge base
                        prompt = f"""Answer this medical question based on the context:

Question: {user_input}

Answer:"""

                        response = model(
                            prompt,
                            max_new_tokens=200,
                            min_new_tokens=50,
                            temperature=0.4,
                            do_sample=True,
                            top_p=0.9,
                            repetition_penalty=2.0
                        )

                        full_answer = response[0]['generated_text']
                        
                        # Extract only the new generated text (remove the prompt)
                        if "Answer:" in full_answer:
                            answer = full_answer.split("Answer:", 1)[1].strip()
                        else:
                            answer = full_answer
                        from_knowledge_base = False
                    
                    answer = clean_response(answer)

                    intent = get_query_intent(user_input)

                    st.success("Healthcare Response:")
                    st.write(answer)

                    # ✅ Smarter summary logic
                    # Only show summary if answer is long AND either from model (not KB) or summary is significantly different
                    if len(answer.split()) > 40:  # Only for longer answers
                        summary = summarizer(answer, max_length=60, min_length=15, do_sample=False)
                        summary_text = summary[0]['summary_text']
                        # Clean up spacing before punctuation
                        summary_text = re.sub(r'\s+([,.!?])', r'\1', summary_text)
                        
                        # Check if summary is meaningfully different from original
                        # If too similar to answer, skip showing it
                        answer_words = set(answer.lower().split())
                        summary_words = set(summary_text.lower().split())
                        similarity = len(answer_words & summary_words) / max(len(answer_words), len(summary_words))
                        
                        # Show summary only if it's from model output or adds new value, but skip for KB definitions
                        if (not from_knowledge_base or similarity < 0.8) and not (from_knowledge_base and intent == "Definition"):
                            st.subheader("📌 Summary")
                            st.write(summary_text)

                except Exception as e:
                    st.error(f"Error: {e}")

        else:
            st.warning("Please enter a question.")

# -------------------------------
# RIGHT SIDE (Medicine Info)
# -------------------------------
with col2:
    st.subheader("💊 Medicine Info")

    med_name = st.text_input("Enter medicine name:")

    if st.button("Get Info"):
        if med_name:
            with st.spinner("Fetching medicine data..."):
                info = get_medicine_info(med_name)

            if info:
                for k, v in info.items():
                    st.write(f"**{k}:** {v}")
            else:
                st.error("No data found.")
        else:
            st.warning("Enter a valid medicine name.")


    
