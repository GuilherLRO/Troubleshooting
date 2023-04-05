import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

def classificate_img(endpoint,event):
    # Decode the image data
    image = base64.b64decode(event["image_data"])

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(
        endpoint_name=ENDPOINT,
        sagemaker_session=sagemaker.Session()
    )

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction:
    inferences = predictor.predict(image)
    
    # We return the data back to the Step Function    
    event["inferences"] = inferences.decode('utf-8')
    
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }