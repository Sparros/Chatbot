
# Initialise Wikipedia agent
import wikipedia
import json, requests
import aiml
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
import pyttsx3
from nltk.sem import Expression
from nltk.inference import ResolutionProver
read_expr = Expression.fromstring

APIkey = "5403a1e0442ce1dd18cb1bf7c40e776f" 

kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

#  Initialise Knowledgebase. 
import pandas
kb=[]
data = pandas.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]
# >>> ADD SOME CODES here for checking KB integrity (no contradiction), 
# otherwise show an error message and terminate




# Read the CSV file and separate questions and answers
questions = []
answers = []
with open('qas.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip header row
    for row in reader:
        questions.append(row[0])
        answers.append(row[1])
lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))

# Define function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered_tokens = [lemmatizer.lemmatize(w) for w in tokens if not w in stopwords]
    return " ".join(filtered_tokens)
# Preprocess all questions and answers
preprocessed_questions = [preprocess_text(q) for q in questions]
preprocessed_answers = [preprocess_text(a) for a in answers]
# Create tf-idf vectorizer for answers
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_answers)

# initialize the text-to-speech engine
engine = pyttsx3.init()
# define a function to speak the response
def speak(text):
    engine.say(text)
    engine.runAndWait()

trigger_phrase = "hey chatbot"

def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        # Recognize speech using Google Speech Recognition
        result = recognizer.recognize_google(audio, show_all=True)
        transcribed_text = result['alternative'][0]['transcript']
        print("You said: " + transcribed_text)
        return transcribed_text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
       print("Could not request results from Google Speech Recognition service; {0}".format(e))
    return ""

#initialize input type as text input
input_type = 'text'

print("Welcome to this chat bot. Please feel free to ask questions from me!")
#speak("Welcome to this chat bot. Please feel free to ask questions from me!")
# Main loop
while True:
    # check if input type is text or speech-to-text
    if input_type == 'text':
        #get user text input
        try:
            userInput = input("> ")
            if userInput.lower() == "audio":
                input_type = 'speech'
                continue
        except (KeyboardInterrupt, EOFError) as e:
            print("Bye!")
            break

    elif input_type == 'speech':
        #get user speech input
        try:
            print("Listening...")
            userInput = recognize_speech_from_mic()
            if userInput == "text input":
                input_type = "text"
                continue
        except (KeyboardInterrupt, EOFError) as e:
            print("Bye!")
            break

    #pre-process user input and determine response agent (if needed)
    # Preprocess user input if not switching input type
    preprocessed_input = preprocess_text(userInput)

    # Search AIML patterns
    response_agent = 'aiml'
    answer = kern.respond(preprocessed_input) 
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        elif cmd == 1:
            try:
                wSummary = wikipedia.summary(params[1], sentences=3,auto_suggest=False)
                print(wSummary)
            except:
                print("Sorry, I do not know that. Be more specific!")
        elif cmd == 2:
            succeeded = False
            api_url = r"http://api.openweathermap.org/data/2.5/weather?q="
            response = requests.get(api_url + params[1] + r"&units=metric&APPID="+APIkey)
            if response.status_code == 200:
                response_json = json.loads(response.content)
                if response_json:
                    t = response_json['main']['temp']
                    tmi = response_json['main']['temp_min']
                    tma = response_json['main']['temp_max']
                    hum = response_json['main']['humidity']
                    wsp = response_json['wind']['speed']
                    wdir = response_json['wind']['deg']
                    conditions = response_json['weather'][0]['description']
                    print("The temperature is", t, "°C, varying between", tmi, "and", tma, "at the moment, humidity is", hum, "%, wind speed ", wsp, "m/s,", conditions)
                    succeeded = True
            if not succeeded:
                print("Sorry, I could not resolve the location you gave me.")
        # Here are the processing of the new logical component:
        elif cmd == 31: # if input pattern is "I know that * is *"
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            # >>> ADD SOME CODES HERE to make sure expr does not contradict 
            # with the KB before appending, otherwise show an error message.
            kb.append(expr) 
            print('OK, I will remember that',object,'is', subject)
        elif cmd == 32: # if the input pattern is "check that * is *"
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            answer=ResolutionProver().prove(expr, kb, verbose=True)
            if answer:
               print('Correct.')
            else:
               print('It may not be true.') 
               # >> This is not an ideal answer.
               # >> ADD SOME CODES HERE to find if expr is false, then give a
               # definite response: either "Incorrect" or "Sorry I don't know." 
        elif cmd == 99:
            # If AIML pattern not found, search Q/A list for most similar question
            # Create tf-idf vector for user input
            input_tfidf = vectorizer.transform([preprocessed_input])
        
            # Calculate cosine similarity between user input and all questions
            similarities = cosine_similarity(input_tfidf, tfidf_matrix)
        
            # Get index of question with highest similarity
            most_similar_idx = similarities.argmax()

            # Get corresponding answer
            most_similar_answer = answers[most_similar_idx]
            print(most_similar_answer)
            # Set answer to most similar answer
            answer = most_similar_answer
            speak(answer)
        else:
            print("I did not get that, please try again.")
    else:
        print(answer)