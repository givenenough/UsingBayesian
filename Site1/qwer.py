import os
os.environ['PATH'] = 'C:/Users/Admin/Downloads/mingw-w64-v10.0.0/mingw64/bin;' + os.environ['PATH']
import theano.tensor as tt
import theano
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
import pymc3 as pm
import requests
import numpy as np

# Define the Scoring Model
def scoring_model(vulnerability_score):
    vulnerability_score = tt.as_tensor_variable(vulnerability_score)
    # Calculate the vulnerability score based on various criteria
    score = 0
    score += tt.maximum(vulnerability_score, 0)
    score += tt.switch(vulnerability_score < -1, 1, 0)
    score += tt.switch(vulnerability_score < -2, 1, 0)
    return score

# Check each link for vulnerabilities
links = ['https://www.example.com', 'https://www.google.com', 'https://www.facebook.com']
vulnerabilities = {}
for link in links:
    # Make a request to the link
    response = requests.get(link)

    # Analyze the response to identify potential vulnerabilities
    vulnerability_score = 0
    if 'password' in response.text:
        vulnerability_score += 1
    if 'sql injection' in response.text:
        vulnerability_score += 2
    if 'xss' in response.text:
        vulnerability_score += 1

    # Cast vulnerability_score to float if necessary
    vulnerability_score = tt.cast(vulnerability_score, 'float32')

    # Assign a score to the vulnerability based on the Scoring Model
    vulnerabilities[link] = scoring_model(vulnerability_score)

# Define the Bayesian Network Model
with pm.Model() as model:
    # Define the variables and their dependencies
    attack_Y_obs = pm.Normal("attack_Y_obs", mu=np.asarray(list(vulnerabilities.values()), dtype=float).flatten(), sigma=1, observed=list(vulnerabilities.values()))

    # Define the Scoring Model as a deterministic variable
    score = pm.Deterministic('score', scoring_model(attack_Y_obs))

    # Define the dependencies between the variables
    vuln_probs = pm.Deterministic("vuln_probs", pm.math.sigmoid(score))
    attack_prob = pm.Deterministic("attack_prob", pm.math.sigmoid(vuln_probs))

    # Calculate the posterior distribution
    trace = pm.sample(1000, tune=1000)

# Print the results
print('Vulnerabilities:', vulnerabilities)
print('Attack probability:', np.mean(trace['attack_prob'] > 0.5))