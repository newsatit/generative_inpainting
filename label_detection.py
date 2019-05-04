import io
from PIL import Image, ImageDraw
import cv2
import numpy as np

def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()
        print(type(content))

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    # print('Texts:')

    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))

def get_text_mask(path, width, height):
    """Detects text in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    my_image = cv2.imread(path)
    my_image = cv2.resize(my_image, (height, width))

    img_str = cv2.imencode('.jpg', my_image)[1].tostring()
    
    image = vision.types.Image(content=img_str)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    img = Image.new('L', (width, height), 0)
    
    for text in texts[1:]:
        
        polygon = ([ (vertex.x,vertex.y) for vertex in text.bounding_poly.vertices])
        ImageDraw.Draw(img).polygon(polygon, outline=255, fill=255)
        
    mask = np.array(img)
    return mask

if __name__ == "__main__":
    detect_text('examples/if-you-think-were-bad-try-looking-in-the-mirror.jpg')