def likelihood(X_train, model, device):
    ##########################################################

    x = X_train.to(device) #shape is [B,D]

    # Fwd pass through the flow to get the log-likelihoods
    log_prob = model.log_prob(x) # Shape: [B,]
    
    # To maximize the log-likelihood, minimize its negative mean, torch minimizes, thats why we need to feed the optimizer with the negative
    loss = -log_prob.mean() # Shape: [B]

    ##########################################################

    return loss
