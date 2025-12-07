import tensorflow as tf
import numpy as np

# Hyperparameters
DEVICE = tf.device("GPU" if tf.config.list_physical_devices('GPU') else "CPU")
MEMORY_CAPACITY = 50000
BATCH_SIZE = 128
LR_A = 1e-7
LR_C = 1e-6
GAMMA = 0.99
TAU = 0.005
DELAY_UPDATE = 2
NOISE_STD = 0.1
NOISE_CLIP = 0.5  # Clip noise amplitude for target action
A_LOSS_CLIP = 1.0  # Gradient clip norm for actor
C_LOSS_CLIP = 1.0  # Gradient clip norm for critic

class Actor(tf.keras.Model):
    def __init__(self, s_dim, a_dim, a_bound):
        super().__init__()
        self.l1 = tf.keras.layers.Dense(128, activation='relu')
        self.l11 = tf.keras.layers.Dense(64, activation='relu')
        self.l2 = tf.keras.layers.Dense(a_dim, activation='tanh')
        self.a_bound = tf.convert_to_tensor(a_bound, dtype=tf.float32)

    def call(self, s):
        x = self.l1(s)
        x = self.l11(x)
        x = self.l2(x)
        return x * self.a_bound

class Critic(tf.keras.Model):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.l1 = tf.keras.layers.Dense(128, activation='relu')
        self.l11 = tf.keras.layers.Dense(64, activation='relu')
        self.l2 = tf.keras.layers.Dense(1)

    def call(self, s, a):
        x = tf.concat([s, a], axis=1)
        x = self.l1(x)
        x = self.l11(x)
        return self.l2(x)

class TD3:
    def __init__(self, a_dim, s_dim, a_bound):
        self.a_dim, self.s_dim = a_dim, s_dim
        self.a_bound = tf.convert_to_tensor(a_bound, dtype=tf.float32)
        
        # Actor networks (online + target)
        self.actor = Actor(s_dim, a_dim, a_bound)
        self.actor_target = Actor(s_dim, a_dim, a_bound)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_A)
        
        # Double Critic networks (online + target)
        self.critic1 = Critic(s_dim, a_dim)
        self.critic2 = Critic(s_dim, a_dim)
        self.critic_target1 = Critic(s_dim, a_dim)
        self.critic_target2 = Critic(s_dim, a_dim)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_C)
        
        # Hard update target networks to match online networks initially
        self.soft_update(self.actor_target, self.actor, tau=1.0)
        self.soft_update(self.critic_target1, self.critic1, tau=1.0)
        self.soft_update(self.critic_target2, self.critic2, tau=1.0)
        
        # Replay buffer
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 2), dtype=np.float32)  # s,a,r,s_,done
        self.pointer = 0
        self.update_count = 0
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def choose_action(self, s, add_noise=True):
        s = tf.convert_to_tensor(s.reshape(1, -1), dtype=tf.float32)
        a = self.actor(s).numpy()[0]
        
        if add_noise:
            noise = np.random.normal(0, NOISE_STD, size=self.a_dim)
            a = np.clip(a + noise, -self.a_bound.numpy(), self.a_bound.numpy())
        return a

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_, [done]))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    @tf.function
    def learn(self):
        # Sample from replay buffer
        sample_size = tf.minimum(self.pointer, MEMORY_CAPACITY)
        indices = tf.random.uniform((BATCH_SIZE,), 0, sample_size, dtype=tf.int32)
        bt = tf.gather(self.memory, indices)
        
        # Parse batch data
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, self.s_dim + self.a_dim: self.s_dim + self.a_dim + 1]
        bs_ = bt[:, self.s_dim + self.a_dim + 1: self.s_dim * 2 + self.a_dim + 1]
        bdone = bt[:, self.s_dim * 2 + self.a_dim + 1:]
        
        # Convert to tensor
        bs = tf.cast(bs, tf.float32)
        ba = tf.cast(ba, tf.float32)
        br = tf.cast(br, tf.float32)
        bs_ = tf.cast(bs_, tf.float32)
        bdone = tf.cast(bdone, tf.float32)

        # Update Critic networks
        with tf.GradientTape() as tape:
            a_target = self.actor_target(bs_)
            noise = tf.random.normal(tf.shape(a_target), 0, NOISE_STD)
            noise = tf.clip_by_value(noise, -NOISE_CLIP, NOISE_CLIP)
            a_target = tf.clip_by_value(a_target + noise, -self.a_bound, self.a_bound)
            
            # Double Q-learning to reduce overestimation
            q_target1 = self.critic_target1(bs_, a_target)
            q_target2 = self.critic_target2(bs_, a_target)
            q_target = tf.minimum(q_target1, q_target2)
            q_target = br + (1 - bdone) * GAMMA * q_target
            
            # Current Q values
            q1 = self.critic1(bs, ba)
            q2 = self.critic2(bs, ba)
            
            # Total critic loss
            critic_loss = self.loss_fn(q_target, q1) + self.loss_fn(q_target, q2)
        
        # Optimize critic
        critic_grads = tape.gradient(critic_loss, self.critic1.trainable_variables + self.critic2.trainable_variables)
        critic_grads = [tf.clip_by_norm(grad, C_LOSS_CLIP) for grad in critic_grads]
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic1.trainable_variables + self.critic2.trainable_variables))

        # Delayed update for Actor network
        self.update_count += 1
        if self.update_count % DELAY_UPDATE == 0:
            with tf.GradientTape() as tape:
                a_current = self.actor(bs)
                actor_loss = -tf.reduce_mean(self.critic1(bs, a_current))
            
            # Optimize actor
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            actor_grads = [tf.clip_by_norm(grad, A_LOSS_CLIP) for grad in actor_grads]
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            
            # Soft update target networks
            self.soft_update(self.actor_target, self.actor, TAU)
            self.soft_update(self.critic_target1, self.critic1, TAU)
            self.soft_update(self.critic_target2, self.critic2, TAU)
            
            self.update_count = 0

    def soft_update(self, target_net, source_net, tau):
        # Target = (1 - tau) * Target + tau * Source
        for target_var, source_var in zip(target_net.trainable_variables, source_net.trainable_variables):
            target_var.assign(target_var * (1.0 - tau) + source_var * tau)

    def save_model(self, file_path):
        # Save network weights
        self.actor.save_weights(f"{file_path}_actor.h5")
        self.critic1.save_weights(f"{file_path}_critic1.h5")
        self.critic2.save_weights(f"{file_path}_critic2.h5")

    def load_model(self, file_path):
        # Load network weights and sync target networks
        self.actor.load_weights(f"{file_path}_actor.h5")
        self.critic1.load_weights(f"{file_path}_critic1.h5")
        self.critic2.load_weights(f"{file_path}_critic2.h5")
        
        self.soft_update(self.actor_target, self.actor, tau=0.9)
        self.soft_update(self.critic_target1, self.critic1, tau=0.9)
        self.soft_update(self.critic_target2, self.critic2, tau=0.9)