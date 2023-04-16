import data
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from sklearn.model_selection import train_test_split
from determined.keras import InputData, TFKerasTrial, TFKerasTrialContext

class ParkinsonsDisease(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext) -> None:
        self.context = context

        # Load Data 
        x, y = data.load_dataset(self.context.get_data_config())
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x, y, test_size=0.40, random_state=1)
        
        # Define Dimensions 
        self.input_shape = x.shape[1]
        self.output_shape = y.shape[1]

    def build_model(self):

        # Define Model 
        model = keras.Sequential(
            [
                keras.layers.Dense(256, input_shape = [self.input_shape], activation = 'relu'),
                keras.layers.Dropout(0.6),
                keras.layers.Dense(256, activation = 'relu'),
                keras.layers.Dropout(0.6),
                keras.layers.Dense(150, activation = 'relu'),
                keras.layers.Dropout(0.6),
                keras.layers.Dense(150, activation = 'relu'),
                keras.layers.Dropout(0.6),
                keras.layers.Dense(128, activation = 'relu'),
                keras.layers.Dropout(0.6),
                keras.layers.Dense(128, activation = 'relu'),
                keras.layers.Dropout(0.6),
                keras.layers.Dense(self.output_shape),
            ]
        )

        # Wrap Model
        model = self.context.wrap_model(model)

        # Create and Wrap Optimizer
        optimizer = tf.keras.optimizers.Adam()
        optimizer = self.context.wrap_optimizer(optimizer)

        # sMAPE Loss Function
        def smape_loss(y_true, y_pred):
            epsilon = 0.1
            numer = K.abs(y_pred - y_true)
            denom = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
            smape = numer / (denom / 2)
            smape = tf.where(tf.math.is_nan(smape), tf.zeros_like(smape), smape)
            return smape

        # Compile Model 
        model.compile(
            optimizer=optimizer,
            loss=smape_loss,
        )
        
        return model

    def build_training_data_loader(self) -> InputData:        
        x_train = self.x_train 
        y_train = self.y_train 

        return x_train, y_train

    def build_validation_data_loader(self) -> InputData:
        x_val = self.x_val 
        y_val = self.y_val

        return x_val, y_val