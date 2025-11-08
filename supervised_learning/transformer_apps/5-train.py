#!/usr/bin/env python3
"""
Trains a Transformer model for Portuguese→English translation.
"""

import tensorflow as tf
import numpy as np
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, dm, warmup_steps=4000):
        super().__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    data = Dataset()

    def tf_encode(pt, en):
        pt_tokens, en_tokens = data.encode(pt, en)
        return pt_tokens, en_tokens

    data_train = data.data_train.map(
        lambda pt, en: tf.py_function(tf_encode, [pt, en], [tf.int64, tf.int64])
    ).padded_batch(batch_size, padded_shapes=([None], [None]))

    data_valid = data.data_valid.map(
        lambda pt, en: tf.py_function(tf_encode, [pt, en], [tf.int64, tf.int64])
    ).padded_batch(batch_size, padded_shapes=([None], [None]))

    transformer = Transformer(N, dm, h, hidden,
                              data.tokenizer_pt.vocab_size + 2,
                              data.tokenizer_en.vocab_size + 2,
                              max_len)

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(y_true, y_pred):
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        loss_ = loss_object(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)

    for epoch in range(1, epochs + 1):
        train_loss.reset_states()
        train_accuracy.reset_states()
        batch_num = 0

        for (batch, (pt, en)) in enumerate(data_train):
            train_step(pt, en)
            batch_num += 1
            if batch_num % 50 == 0 or batch_num == 1:
                print(f"Epoch {epoch}, Batch {batch_num}: "
                      f"Loss {train_loss.result().numpy()}, "
                      f"Accuracy {train_accuracy.result().numpy()}")

        print(f"Epoch {epoch}: Loss {train_loss.result().numpy()}, "
              f"Accuracy {train_accuracy.result().numpy()}")

    return transformer
