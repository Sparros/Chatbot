from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import time
import requests, uuid


def imageTextTranslation(language):
    '''
    Authenticate
    Authenticates your credentials and creates a client.
    '''
    subscription_key = "7615f92a70e8474eb569b19d19cc29aa"
    endpoint = "https://chatbot-translation.cognitiveservices.azure.com/"
    translate_endpoint = "https://api.cognitive.microsofttranslator.com/"
    location = "uksouth"

    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
    '''
    END - Authenticate
    '''
    textToTranslate = ""

    language_dict = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Chinese": "zh",
        "Japanese": "ja",
        "Korean": "ko",
        "Russian": "ru",
        "Arabic": "ar"
    }

    '''
    OCR: Read File using the Read API, extract text - remote
    This example will extract text in an image, then print results, line by line.
    This API call can also extract handwriting style text (not shown).
    '''
    print("Select option:\n 1. URL\n 2. Local Image")
    user_input = input("> ")
    if user_input == "1":
        # Get an image with text from URL
        user_input = input("Enter image url: ")
        read_image_url = user_input
        
        # Call API with URL and raw response (allows you to get the operation location)
        read_response = computervision_client.read(read_image_url, raw=True)   
    elif user_input == "2":
        # Get an image with text from local file
        user_input = input("Enter image path: ")
        read_image_path = r"{}".format(user_input)       
        
        # Call API with file path and raw response (allows you to get the operation location)
        with open(read_image_path, "rb") as image_stream:
            read_response = computervision_client.read_in_stream(image_stream, raw=True)
    else:
        print("Invalid option selected")
        exit()

    # Get the operation location (URL with an ID at the end) from the response
    read_operation_location = read_response.headers["Operation-Location"]
    # Grab the ID from the URL
    operation_id = read_operation_location.split("/")[-1]

    # Call the "GET" API and wait for it to retrieve the results 
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    # Print the detected text, line by line
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                textToTranslate += line.text + " "
                print(line.text)
                #print(line.bounding_box)

    #target_language = input("> ").capitalize()
    language_code = language_dict[language.capitalize()]
    path = 'translate'
    constructed_url = translate_endpoint + path

    params = {
        'api-version': '3.0',
        'to': language_code
    }

    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        # location required if you're using a multi-service or regional (not global) resource.
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # You can pass more than one object in body.
    body = [{
        'text': textToTranslate
    }]

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()

    print(response[0]['translations'][0]['text'])
    #print(json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': ')))

imageTextTranslation()