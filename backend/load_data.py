from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an image data generator with preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,                # Normalize pixel values
    rotation_range=10,             # Rotate images up to 10 degrees
    width_shift_range=0.1,         # Shift width by 10%
    height_shift_range=0.1,        # Shift height by 10%
    horizontal_flip=True,          # Flip images horizontally
    validation_split=0.2           # Split training data into train/validation
)

# Load training images
train_generator = datagen.flow_from_directory(
    "../dataset/train",
    target_size=(128, 128),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=32,
    subset="training"
)

# Load validation images
val_generator = datagen.flow_from_directory(
    "../dataset/train",
    target_size=(128, 128),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=32,
    subset="validation"
)

# Load test images
test_generator = datagen.flow_from_directory(
    "../dataset/test",
    target_size=(128, 128),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=32,
    shuffle=False
)