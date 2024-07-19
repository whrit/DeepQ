import tensorflow as tf
from datetime import datetime

class TensorboardLogger:
    def __init__(self, log_dir):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = f'{log_dir}/train_{current_time}'
        self.val_log_dir = f'{log_dir}/val_{current_time}'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)

    def log_train(self, episode, loss, total_profit, epsilon):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('Loss', loss, step=episode)
            tf.summary.scalar('Total Profit', total_profit, step=episode)
            tf.summary.scalar('Epsilon', epsilon, step=episode)

    def log_validation(self, episode, total_profit):
        with self.val_summary_writer.as_default():
            tf.summary.scalar('Validation Total Profit', total_profit, step=episode)

    def log_action_distribution(self, episode, action_counts):
        with self.train_summary_writer.as_default():
            for action, count in action_counts.items():
                tf.summary.scalar(f'Action_{action}_Count', count, step=episode)

    def log_custom_metric(self, name, value, step, is_train=True):
        writer = self.train_summary_writer if is_train else self.val_summary_writer
        with writer.as_default():
            tf.summary.scalar(name, value, step=step)