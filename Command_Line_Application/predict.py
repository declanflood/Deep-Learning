import tensorflow as tf
import tensorflow_hub as hub
import argparse
from PIL import Image
import numpy as np
import json as json

image_size = 224 # Must use this size as MobileNet was trained on images this size

# =============================================================================
# Subroutines
# =============================================================================

# Function to parse the arguments passed through the command line
def arg_parser():
    parser = argparse.ArgumentParser(description="Determines the type of flower in an image.")

    parser.add_argument('image', 
                        type=str, 
                        help='Point to input flower impage file.')

    parser.add_argument('saved_model', 
                        type=str, 
                        help='Point to saved model.')
    
    parser.add_argument('--top_k', 
                        type=int,
                        default=1,
                        help='Choose top K matches.')
    
    parser.add_argument('--category_names', 
                        type=str, 
                        default='None',
                        help='Path to a JSON file mapping labels to flower names.')

    args = parser.parse_args()
    
    return args



# Load a previously trained model into memory
def load_model(model_name):
    loaded_model = tf.keras.models.load_model(model_name,
                                              custom_objects={'KerasLayer': hub.KerasLayer})
    return loaded_model



# Load, resize and normalise image as preparation for inference
def load_process_image(image_path):    
    im = Image.open(image_path)
    image = np.asarray(im)
    image = np.expand_dims(image, axis=0)    
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255 
    return image.numpy()


# predict flower in processess_test_image using model and return best matching top_k indexes
def predict(processed_test_image, model, top_k):
    prediction = model.predict(processed_test_image)
    top_values, top_indices = tf.math.top_k(prediction, k=top_k, sorted=True)
    return top_indices



# =============================================================================
# Main Function
#
# Sample usage: python predict.py ./test_images/wild_pansy.jpg temp_model.h5 --category_names label_map.json --top_k 3
# =============================================================================
def main():  
    args = arg_parser()
    
    model = load_model(args.saved_model)
    
    image = load_process_image(args.image)
   
    top_indices = predict(image, model, args.top_k)
    
    if args.category_names=='None':
        # simply print the top_k indexes
        print('Flower index prediction', top_indices.numpy())
    else:
        # load a json file which converts indexes into flower names
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)        
        # Must add one because model indexes start range from 0 - 101, while json labels are 1 - 102
        top_classes = [class_names[str(value+1)] for value in top_indices.numpy()[0]]
        print('Flower name prediction:', top_classes)
        


# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()



