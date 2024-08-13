from My_Functions import train_data, test_data
import tensorflow as tf
models = tf.keras.models
layers = tf.keras.layers
Model = tf.keras.Model

def create_model(x_train, y_train, x_test, y_test):
    # Input Layer
    input_layer = layers.Input(shape=(283))

    # Hidden Layer
    dense_layer_1 = layers.Dense(60)(input_layer)
    dense_layer_1 = layers.Activation('relu')(dense_layer_1)

    dense_layer_2 = layers.Dense(40)(dense_layer_1)
    dense_layer_2 = layers.Activation('relu')(dense_layer_2)

    dense_layer_3 = layers.Dense(30)(dense_layer_2)
    dense_layer_3 = layers.Activation('relu')(dense_layer_3)

    # Dropout Layer
    dropout_layer = layers.Dropout(0.3)(dense_layer_3)

    # Output Layer
    output_layer = layers.Dense(5)(dropout_layer)
    output_layer = layers.Activation('softmax')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    # Training
    model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=15, epochs=100, verbose=1)
    result = model.evaluate(x_test, y_test, verbose=0)
    return result


# -------------------------------------------------------------------------------------------------
# training & testing Data processing
if __name__ == '__main__':
    sampleRate = 20000
    x_train, y_train = train_data(sampleRate)
    x_test, y_test = test_data(sampleRate)
    result = create_model(x_train, y_train, x_test, y_test)
    print('Test loss:', result[0])
    print('Test accuracy:', result[1])
