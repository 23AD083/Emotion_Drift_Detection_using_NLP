
import gradio as gr
import torch
import pickle
import plotly.graph_objects as go
import torch.nn.functional as F

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


# ----------------------------
# LOAD MODEL
# ----------------------------

tokenizer = DistilBertTokenizer.from_pretrained("emotion_model")
model = DistilBertForSequenceClassification.from_pretrained("emotion_model")

with open("label_encoder.pkl","rb") as f:
    le = pickle.load(f)


emotion_score = {
    "joy":3,
    "surprise":2,
    "neutral":1,
    "sadness":-1,
    "fear":-2,
    "anger":-3
}


# ----------------------------
# EMOTION PREDICTION
# ----------------------------

def predict_emotion(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)

    pred = torch.argmax(probs).item()

    confidence = torch.max(probs).item()

    emotion = le.inverse_transform([pred])[0]

    return emotion, confidence


# ----------------------------
# DRIFT DETECTION
# ----------------------------

def detect_drift(emotions):

    drift = []

    for i in range(1,len(emotions)):

        prev = emotion_score.get(emotions[i-1],0)
        curr = emotion_score.get(emotions[i],0)

        if abs(curr-prev) >= 2:
            drift.append(i+1)

    return drift


# ----------------------------
# MAIN ANALYSIS FUNCTION
# ----------------------------

def analyze_conversation(conversation):

    lines = conversation.split("
")

    emotions = []
    confidence = []

    for line in lines:

        if line.strip() != "":
            emo, conf = predict_emotion(line)

            emotions.append(emo)
            confidence.append(round(conf*100,2))


    drift = detect_drift(emotions)

    scores = [emotion_score[e] for e in emotions]


    # Plot graph
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(1,len(scores)+1)),
        y=scores,
        mode="lines+markers",
        name="Emotion Trend"
    ))


    # Mental Health Analysis
    negative_count = sum(
        1 for e in emotions if e in ["sadness","fear","anger"]
    )

    if len(emotions) > 3 and negative_count >= len(emotions)/2:

        mental_status = '''
WARNING: Mental Health Alert

Your conversation shows a strong negative emotional trend.

Suggestions:
- Take a short break
- Talk to a trusted friend
- Try relaxation techniques
'''

    else:
        mental_status = "SUCCESS: Emotional state appears stable."


    return emotions, confidence, drift, fig, mental_status


# ----------------------------
# SAMPLE INPUTS
# ----------------------------

samples = {
"Happy → Angry Drift":
'''I was very happy today
My code worked perfectly
Suddenly my laptop crashed
Now I feel very angry''',

"Angry → Happy Recovery":
'''My project was not working
I felt very frustrated
My friend helped me debug
Now everything works
Now I feel very happy''',

"Stable Positive Mood":
'''Today was a great day
I enjoyed working on my project
Everything went smoothly
I feel satisfied''',

"Mental Stress Pattern":
'''I feel very sad today
Nothing seems to work
I feel anxious about my future
Everything feels overwhelming''',

"Mixed Emotion Conversation":
'''I was excited to start my project
But I faced many bugs
I got very angry and frustrated
Later I solved the issue
Now I feel relieved'''
}


# ----------------------------
# GRADIO UI
# ----------------------------

with gr.Blocks(title="Emotion Drift Detection") as demo:

    gr.Markdown("# Emotion Drift Detection Chatbot")

    gr.Markdown("Analyze emotional patterns and detect emotional drift in conversations.")

    sample_choice = gr.Dropdown(
        list(samples.keys()),
        label="Choose Sample Test"
    )

    sample_text = gr.Textbox(
        label="Sample Conversation",
        lines=6
    )

    sample_choice.change(
        lambda x: samples[x],
        sample_choice,
        sample_text
    )


    conversation_input = gr.Textbox(
        label="Enter Conversation (one sentence per line)",
        lines=8
    )

    analyze_btn = gr.Button("Analyze Emotion")


    emotion_output = gr.JSON(label="Predicted Emotions")

    confidence_output = gr.JSON(label="Confidence (%)")

    drift_output = gr.JSON(label="Emotion Drift Points")

    graph_output = gr.Plot(label="Emotion Trend Graph")

    mental_output = gr.Textbox(label="Mental Health Analysis")


    analyze_btn.click(
        analyze_conversation,
        inputs=conversation_input,
        outputs=[
            emotion_output,
            confidence_output,
            drift_output,
            graph_output,
            mental_output
        ]
    )


# ----------------------------
# LAUNCH
# ----------------------------

demo.launch()
