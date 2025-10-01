import numpy as np
import os
import cv2
import tensorflow as tf
import skimage.color as color
import skimage.io as io
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

# ==== CONFIG ====
train_dir = r'C:\Users\Ayub\.cache\kagglehub\datasets\aayush9753\image-colorization-dataset\versions\1\data\train_color'
test_dir = r'C:\Users\Ayub\.cache\kagglehub\datasets\aayush9753\image-colorization-dataset\versions\1\data\test_color'
save_dir = os.path.join(os.getcwd(), "colorization_outputs")
os.makedirs(save_dir, exist_ok=True)

# ==== LOAD DATA ====
def load_and_preprocess_images(directory, max_images=None):
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if max_images:
        files = files[:max_images]
    N = len(files)
    data = np.zeros([N, 256, 256, 3])

    for i, f in enumerate(files):
        img = io.imread(os.path.join(directory, f))
        if img.ndim == 2:  # if grayscale, convert to RGB
            img = np.stack([img]*3, axis=-1)
        img = cv2.resize(img, (256, 256))
        data[i] = img

    # Convert to LAB
    lab_data = color.rgb2lab(data / 255.0)
    X_L = lab_data[:, :, :, 0] / 100.0  # L in [0,1]
    Y_ab = (lab_data[:, :, :, 1:] + 128) / 255.0  # ab in [0,1]
    return X_L[..., np.newaxis], Y_ab, files

# Load data
X_train, Y_train, train_files = load_and_preprocess_images(train_dir, max_images=2000)
X_test, Y_test, test_files = load_and_preprocess_images(test_dir, max_images=200)

print(f"Training data: {X_train.shape[0]} images")
print(f"Test data: {X_test.shape[0]} images")

# ==== MODEL ARCHITECTURE ====
def create_weights(shape, name):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1), name=name)

def create_bias(size, name):
    return tf.Variable(tf.constant(0.1, shape=[size]), name=name)

def convolution(inputs, in_channels, f_size, out_channels, name):
    with tf.compat.v1.variable_scope(name):
        W = create_weights([f_size, f_size, in_channels, out_channels], 'weights')
        b = create_bias(out_channels, 'bias')
        conv = tf.nn.conv2d(inputs, W, strides=[1,1,1,1], padding='SAME') + b
        # Add batch normalization
        mean, variance = tf.nn.moments(conv, axes=[0,1,2])
        conv = tf.nn.batch_normalization(conv, mean, variance, None, None, 1e-5)
        return tf.nn.relu(conv)

def maxpool(inputs, name):
    return tf.nn.max_pool2d(inputs, ksize=2, strides=2, padding='SAME', name=name)

def upsample(inputs, name):
    with tf.compat.v1.variable_scope(name):
        shape = tf.shape(inputs)
        return tf.image.resize(inputs, [shape[1]*2, shape[2]*2], method='nearest')

# Build the model
tf.compat.v1.reset_default_graph()
x = tf.compat.v1.placeholder(tf.float32, [None, 256, 256, 1], name='input_L')
ytrue = tf.compat.v1.placeholder(tf.float32, [None, 256, 256, 2], name='true_ab')

# Encoder
with tf.compat.v1.variable_scope('encoder'):
    conv1 = convolution(x, 1, 3, 64, 'conv1')
    pool1 = maxpool(conv1, 'pool1')
    conv2 = convolution(pool1, 64, 3, 128, 'conv2')
    pool2 = maxpool(conv2, 'pool2')
    conv3 = convolution(pool2, 128, 3, 256, 'conv3')
    pool3 = maxpool(conv3, 'pool3')
    conv4 = convolution(pool3, 256, 3, 512, 'conv4')
    pool4 = maxpool(conv4, 'pool4')
    conv5 = convolution(pool4, 512, 3, 512, 'conv5')
    pool5 = maxpool(conv5, 'pool5')

# Decoder
with tf.compat.v1.variable_scope('decoder'):
    up1 = upsample(pool5, 'up1')
    conv6 = convolution(up1, 512, 3, 512, 'conv6')
    up2 = upsample(conv6, 'up2')
    conv7 = convolution(up2, 512, 3, 256, 'conv7')
    up3 = upsample(conv7, 'up3')
    conv8 = convolution(up3, 256, 3, 128, 'conv8')
    up4 = upsample(conv8, 'up4')
    conv9 = convolution(up4, 128, 3, 64, 'conv9')
    up5 = upsample(conv9, 'up5')
    conv10 = convolution(up5, 64, 3, 32, 'conv10')

    # Final output with constrained activation
    W_out = create_weights([3, 3, 32, 2], 'weights_out')
    b_out = create_bias(2, 'bias_out')
    output_layer = tf.nn.conv2d(conv10, W_out, strides=[1,1,1,1], padding='SAME') + b_out
    output = tf.nn.tanh(output_layer) * 0.5 + 0.5  # Constrain to [0,1]

# ==== LOSS FUNCTION ====
def lab_loss(y_true, y_pred):
    # L1 loss
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Penalize extreme AB values
    ab_penalty = tf.reduce_mean(tf.square(y_pred - 0.5))
    
    return l1_loss + 0.1 * ab_penalty

loss = lab_loss(ytrue, output)

# ==== OPTIMIZER WITH GRADIENT CLIPPING ====
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=5e-4)
grads_and_vars = optimizer.compute_gradients(loss)
capped_grads_and_vars = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grads_and_vars]
train_op = optimizer.apply_gradients(capped_grads_and_vars)

# ==== UTILITY FUNCTIONS ====
def safe_lab2rgb(L, ab):
    """Convert LAB to RGB with clipping to valid ranges"""
    lab = np.zeros((L.shape[0], L.shape[1], 3))
    lab[:, :, 0] = np.clip(L * 100.0, 0, 100)       # L in [0, 100]
    lab[:, :, 1:] = np.clip(ab * 255.0 - 128, -128, 127)  # AB in [-128, 127]
    return np.clip(color.lab2rgb(lab), 0, 1)

def save_colorization_results(input_L, output_ab, true_ab, filename, epoch, save_dir):
    """Save comparison of input grayscale, predicted color, and true color"""
    # Convert using safe LAB conversion
    rgb_pred = safe_lab2rgb(input_L, output_ab)
    rgb_true = safe_lab2rgb(input_L, true_ab)
    rgb_gray = np.repeat(input_L[..., np.newaxis], 3, axis=-1)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb_gray)
    axes[0].set_title('Input Grayscale')
    axes[0].axis('off')
    
    axes[1].imshow(rgb_pred)
    axes[1].set_title('Predicted Color')
    axes[1].axis('off')
    
    axes[2].imshow(rgb_true)
    axes[2].set_title('True Color')
    axes[2].axis('off')
    
    plt.suptitle(f'Epoch {epoch+1}')
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, f'epoch_{epoch+1:03d}_{os.path.splitext(filename)[0]}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()

def evaluate_test_set(session, X_test, Y_test, batch_size):
    """Evaluate model on full test set"""
    total_loss = 0.0
    num_batches = int(np.ceil(X_test.shape[0] / batch_size))
    
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, X_test.shape[0])
        batch_loss = session.run(loss, feed_dict={
            x: X_test[start:end],
            ytrue: Y_test[start:end]
        })
        total_loss += batch_loss * (end - start)
    
    return total_loss / X_test.shape[0]

# ==== TRAINING LOOP ====
batch_size = 16
epochs = 100  # Increased epochs for better convergence
num_samples = X_train.shape[0]
steps_per_epoch = num_samples // batch_size

with tf.compat.v1.Session() as session:
    session.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    best_test_loss = float('inf')

    # Save reference images
    for idx in [0, 1, 2]:
        save_colorization_results(
            np.squeeze(X_test[idx]), 
            np.zeros_like(np.squeeze(Y_test[idx])), 
            np.squeeze(Y_test[idx]), 
            f"reference_{test_files[idx]}", 
            -1, 
            save_dir
        )

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(num_samples)
        X_train, Y_train = X_train[indices], Y_train[indices]
        train_files = [train_files[i] for i in indices]

        # Training
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            start = step * batch_size
            end = start + batch_size
            
            _, batch_loss = session.run([train_op, loss], feed_dict={
                x: X_train[start:end],
                ytrue: Y_train[start:end]
            })
            epoch_loss += batch_loss
        
        epoch_loss /= steps_per_epoch
        
        # Evaluation
        test_loss = evaluate_test_set(session, X_test, Y_test, batch_size)
        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {epoch_loss:.5f}, Test Loss = {test_loss:.5f}")

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            saver.save(session, os.path.join(save_dir, 'best_model.ckpt'))
        
        # Visualizations
        random_idx = np.random.randint(0, X_test.shape[0])
        sample_L = np.squeeze(X_test[random_idx])
        sample_ab_pred = np.squeeze(session.run(output, feed_dict={x: X_test[random_idx:random_idx+1]}))
        save_colorization_results(
            sample_L,
            sample_ab_pred,
            np.squeeze(Y_test[random_idx]),
            f"random_{test_files[random_idx]}", 
            epoch, 
            save_dir
        )
        
        # Save fixed test images
        for idx in [0, 1, 2]:
            fixed_pred = np.squeeze(session.run(output, feed_dict={x: X_test[idx:idx+1]}))
            save_colorization_results(
                np.squeeze(X_test[idx]),
                fixed_pred,
                np.squeeze(Y_test[idx]),
                f"fixed_{idx}_{test_files[idx]}", 
                epoch, 
                save_dir
            )

    print("Training complete!")