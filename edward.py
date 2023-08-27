from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
%matplotlib inline
import csv
import IPython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import tensorflow_probability as tfp
import warnings

from tensorflow_probability import edward2 as ed

plt.style.use('ggplot')


import os
os.chdir(r'C:\Users\mhosseini\Desktop\Python app')

import pandas

with open('InstEval.csv', 'r') as csvfile:
    iterator = csv.reader(csvfile)
    columns = next(iterator)[1:]
    x_train = np.array([row[1:] for row in iterator], dtype=np.int)
    metadata = {'columns': columns}




data, metadata = x_train, metadata
data = pd.DataFrame(data, columns=metadata['columns'])
data = data.rename(columns={'s': 'students',
                            'd': 'instructors',
                            'dept': 'departments',
                            'y': 'ratings'})
data['students'] -= 1  # start index by 0
# Remap categories to start from 0 and end at max(category).
data['instructors'] = data['instructors'].astype('category').cat.codes
data['departments'] = data['departments'].astype('category').cat.codes

train = data.sample(frac=0.8)
test = data.drop(train.index)

train.head()


get_value = lambda dataframe, key, dtype: dataframe[key].values.astype(dtype)
features_train = {
    k: get_value(train, key=k, dtype=np.int32)
    for k in ['students', 'instructors', 'departments', 'service']}
labels_train = get_value(train, key='ratings', dtype=np.float32)

features_test = {k: get_value(test, key=k, dtype=np.int32)
                 for k in ['students', 'instructors', 'departments', 'service']}
labels_test = get_value(test, key='ratings', dtype=np.float32)

num_students = max(features_train['students']) + 1
num_instructors = max(features_train['instructors']) + 1
num_departments = max(features_train['departments']) + 1
num_observations = train.shape[0]

print("Number of students:", num_students)
print("Number of instructors:", num_instructors)
print("Number of departments:", num_departments)
print("Number of observations:", num_observations)

def linear_mixed_effects_model(features):
  # Set up fixed effects and other parameters.
  intercept = tf.get_variable("intercept", [])  # alpha in eq
  effect_service = tf.get_variable("effect_service", [])  # beta in eq
  stddev_students = tf.exp(
      tf.get_variable("stddev_unconstrained_students", []))  # sigma in eq
  stddev_instructors = tf.exp(
      tf.get_variable("stddev_unconstrained_instructors", [])) # sigma in eq
  stddev_departments = tf.exp(
      tf.get_variable("stddev_unconstrained_departments", [])) # sigma in eq

  # Set up random effects.
  effect_students = ed.MultivariateNormalDiag(
      loc=tf.zeros(num_students),
      scale_identity_multiplier=stddev_students,
      name="effect_students")
  effect_instructors = ed.MultivariateNormalDiag(
      loc=tf.zeros(num_instructors),
      scale_identity_multiplier=stddev_instructors,
      name="effect_instructors")
  effect_departments = ed.MultivariateNormalDiag(
      loc=tf.zeros(num_departments),
      scale_identity_multiplier=stddev_departments,
      name="effect_departments")

  # Set up likelihood given fixed and random effects.
  # Note we use `tf.gather` instead of matrix-multiplying a design matrix of
  # one-hot vectors. The latter is memory-intensive if there are many groups.
  ratings = ed.Normal(
      loc=(effect_service * features["service"] +
           tf.gather(effect_students, features["students"]) +
           tf.gather(effect_instructors, features["instructors"]) +
           tf.gather(effect_departments, features["departments"]) +
           intercept),
      scale=1.,
      name="ratings")
  return ratings

# Wrap model in a template. All calls to the model template will use the same
# TensorFlow variables.
model_template = tf.make_template("model", linear_mixed_effects_model)

def linear_mixed_effects_model(features):
  # Set up fixed effects and other parameters.
  intercept = tf.get_variable("intercept", [])  # alpha in eq
  effect_service = tf.get_variable("effect_service", [])  # beta in eq
  stddev_students = tf.exp(
      tf.get_variable("stddev_unconstrained_students", []))  # sigma in eq
  stddev_instructors = tf.exp(
      tf.get_variable("stddev_unconstrained_instructors", [])) # sigma in eq
  stddev_departments = tf.exp(
      tf.get_variable("stddev_unconstrained_departments", [])) # sigma in eq

  # Set up random effects.
  effect_students = ed.MultivariateNormalDiag(
      loc=tf.zeros(num_students),
      scale_identity_multiplier=stddev_students,
      name="effect_students")
  effect_instructors = ed.MultivariateNormalDiag(
      loc=tf.zeros(num_instructors),
      scale_identity_multiplier=stddev_instructors,
      name="effect_instructors")
  effect_departments = ed.MultivariateNormalDiag(
      loc=tf.zeros(num_departments),
      scale_identity_multiplier=stddev_departments,
      name="effect_departments")

  # Set up likelihood given fixed and random effects.
  # Note we use `tf.gather` instead of matrix-multiplying a design matrix of
  # one-hot vectors. The latter is memory-intensive if there are many groups.
  ratings = ed.Normal(
      loc=(effect_service * features["service"] +
           tf.gather(effect_students, features["students"]) +
           tf.gather(effect_instructors, features["instructors"]) +
           tf.gather(effect_departments, features["departments"]) +
           intercept),
      scale=1.,
      name="ratings")
  return ratings

# Wrap model in a template. All calls to the model template will use the same
# TensorFlow variables.
model_template = tf.make_template("model", linear_mixed_effects_model)



log_joint = ed.make_log_joint_fn(model_template)

def target_log_prob_fn(effect_students, effect_instructors, effect_departments):
  """Unnormalized target density as a function of states."""
  return log_joint(  # fix `features` and `ratings` to the training data
    features=features_train,
    effect_students=effect_students,
    effect_instructors=effect_instructors,
    effect_departments=effect_departments,
    ratings=labels_train)

tf.reset_default_graph()

# Set up E-step (MCMC).
effect_students = tf.get_variable(  # `trainable=False` so unaffected by M-step
    "effect_students", [num_students], trainable=False)
effect_instructors = tf.get_variable(
    "effect_instructors", [num_instructors], trainable=False)
effect_departments = tf.get_variable(
    "effect_departments", [num_departments], trainable=False)

hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target_log_prob_fn,
    step_size=0.015,
    num_leapfrog_steps=3)

current_state = [effect_students, effect_instructors, effect_departments]
next_state, kernel_results = hmc.one_step(
      current_state=current_state,
      previous_kernel_results=hmc.bootstrap_results(current_state))

expectation_update = tf.group(
    effect_students.assign(next_state[0]),
    effect_instructors.assign(next_state[1]),
    effect_departments.assign(next_state[2]))

# Set up M-step (gradient descent).
# The following should work. However, TensorFlow raises an error about taking
# gradients through IndexedSlices tensors. This may be a TF bug. For now,
# we recompute the target's log probability at the current state.
# loss = -kernel_results.accepted_results.target_log_prob
with tf.control_dependencies([expectation_update]):
  loss = -target_log_prob_fn(effect_students,
                             effect_instructors,
                             effect_departments)
  optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
  minimization_update = optimizer.minimize(loss)

